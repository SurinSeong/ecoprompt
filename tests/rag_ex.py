import os
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 설정 상수
CONFIG = {
    "MODEL_NAME": "gpt-4o-mini",
    "EMBEDDING_MODEL": "text-embedding-3-small",
    "CHUNK_SIZE": 500,
    "CHUNK_OVERLAP": 50,
    "TOP_K": 4,  # 검색할 문서 개수
    "TEMPERATURE": 0,
    "PDF_PATH": "./data/SPRi AI Brief_Special_AI 에이전트_241209_F.pdf" # 여러분이 원하는 문서를 Data 디렉토리를 만들어서 넣으세요
}


def check_api_key() -> bool:
    """API 키 확인"""
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY가 설정되지 않았습니다!")
        print("\n해결 방법:")
        print("1. .env 파일 생성 후 다음 내용 추가:")
        print("   OPENAI_API_KEY=your-api-key-here")
        print("\n2. 또는 터미널에서:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return False
    return True


def check_pdf_exists(pdf_path: str) -> bool:
    """PDF 파일 존재 확인"""
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF 파일을 찾을 수 없습니다: {pdf_path}")
        print("\n해결 방법:")
        print("1. data/ 폴더를 생성하세요")
        print("2. PDF 파일을 data/ 폴더에 넣으세요")
        print("3. 또는 CONFIG['PDF_PATH'] 를 수정하세요")
        return False
    return True


# ==========================================
# 일반 챗봇 (RAG 없이)
# ==========================================
def setup_general_openai():
    """일반 OpenAI 클라이언트 설정"""
    from openai import OpenAI
    
    try:
        client = OpenAI()
        return client
    except Exception as e:
        print(f"[ERROR] OpenAI 클라이언트 초기화 실패: {e}")
        return None


def general_openai_chat(client, question: str) -> tuple[str, float]:
    """
    일반 챗봇: 학습된 지식만으로 답변
    Returns: (답변, 소요시간)
    """
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=CONFIG["MODEL_NAME"],
            messages=[
                {
                    "role": "system",
                    "content": "당신은 도움이 되는 AI 어시스턴트입니다. 정확한 정보만 제공하고, 모르는 내용은 솔직히 모른다고 답변하세요."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=CONFIG["TEMPERATURE"]
        )
        elapsed_time = time.time() - start_time
        return response.choices[0].message.content, elapsed_time
    except Exception as e:
        return f"[ERROR] 오류 발생: {str(e)}", 0.0


# ==========================================
# RAG 기반 챗봇
# ==========================================
def setup_rag_system(pdf_path: str):
    """RAG 시스템 전체 설정"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    print("\n" + "=" * 80)
    print("RAG 시스템 초기화 중...")
    print("=" * 80)
    
    try:
        # Step 1: Document Loader
        print("\n[1/7] Document Loader - 문서를 로드합니다...")
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        print(f"      ✓ 로드된 문서 페이지 수: {len(docs)}")
        
        # Step 2: Text Splitter
        print("\n[2/7] Text Splitter - 문서를 청크로 분할합니다...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["CHUNK_SIZE"],
            chunk_overlap=CONFIG["CHUNK_OVERLAP"]
        )
        split_documents = text_splitter.split_documents(docs)
        print(f"      ✓ 분할된 청크 수: {len(split_documents)}")
        
        # Step 3: Embedding
        print("\n[3/7] Embedding - 텍스트를 벡터로 변환합니다...")
        embeddings = OpenAIEmbeddings(model=CONFIG["EMBEDDING_MODEL"])
        print(f"      ✓ 임베딩 모델: {CONFIG['EMBEDDING_MODEL']}")
        
        # Step 4: Vector Store
        print("\n[4/7] Vector Store - FAISS 벡터 DB를 생성합니다...")
        vectorstore = FAISS.from_documents(
            documents=split_documents,
            embedding=embeddings
        )
        print(f"      ✓ Vector Store 생성 완료")
        
        # Step 5: Retriever
        print("\n[5/7] Retriever - 검색기를 설정합니다...")
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": CONFIG["TOP_K"]}
        )
        print(f"      ✓ Top-K: {CONFIG['TOP_K']}개 문서 검색")
        
        # Step 6: Prompt
        print("\n[6/7] Prompt - 프롬프트 템플릿을 구성합니다...")
        prompt = PromptTemplate.from_template(
            """당신은 문서 기반 질의응답 전문 AI 어시스턴트입니다.

[지침]
1. 제공된 문서 내용만을 바탕으로 답변하세요
2. 문서에 없는 내용은 추측하지 말고 "문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요
3. 답변은 명확하고 구체적으로 작성하세요
4. 가능하면 문서의 어느 부분에서 정보를 얻었는지 언급하세요

[검색된 문서 내용]
{context}

[사용자 질문]
{question}

[답변]"""
        )
        print("      ✓ 프롬프트 템플릿 준비 완료")
        
        # Step 7: LLM
        print("\n[7/7] LLM - 언어 모델을 설정합니다...")
        llm = ChatOpenAI(
            model_name=CONFIG["MODEL_NAME"],
            temperature=CONFIG["TEMPERATURE"]
        )
        print(f"      ✓ 모델: {CONFIG['MODEL_NAME']}")
        
        # Step 8: Chain
        print("\n[8/8] RAG Chain - 파이프라인을 연결합니다...")
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("      ✓ RAG Chain 구성 완료")
        
        print("\n" + "=" * 80)
        print("✓ RAG 시스템 초기화 완료!")
        print("=" * 80)
        
        return {
            "chain": rag_chain,
            "retriever": retriever,
            "vectorstore": vectorstore,
            "split_documents": split_documents
        }
        
    except Exception as e:
        print(f"\n[ERROR] RAG 시스템 초기화 실패: {str(e)}")
        return None


def rag_chat(rag_chain, question: str) -> tuple[str, float]:
    """
    RAG 기반 답변 생성
    Returns: (답변, 소요시간)
    """
    try:
        start_time = time.time()
        answer = rag_chain.invoke(question)
        elapsed_time = time.time() - start_time
        return answer, elapsed_time
    except Exception as e:
        return f"[ERROR] 오류 발생: {str(e)}", 0.0


def get_retrieved_docs(retriever, question: str) -> List:
    """RAG에서 검색된 문서 가져오기"""
    try:
        docs = retriever.invoke(question)
        return docs
    except Exception as e:
        print(f"[ERROR] 문서 검색 중 오류: {str(e)}")
        return []


def display_retrieved_docs(docs: List, max_chars: int = 200):
    """검색된 문서를 보기 좋게 출력"""
    if not docs:
        print("\n    검색된 문서가 없습니다.")
        return
    
    print(f"\n참조 문서 ({len(docs)}개):")
    print("  " + "─" * 76)
    
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get('page', 'N/A')
        if isinstance(page, int):
            page = page + 1  # 0-based를 1-based로 변환
        
        source = os.path.basename(doc.metadata.get('source', 'N/A'))
        content = doc.page_content.strip()
        
        # 내용이 너무 길면 축약
        if len(content) > max_chars:
            content_preview = content[:max_chars] + "..."
        else:
            content_preview = content
        
        print(f"\n  [{i}] 페이지 {page} | {source}")
        # 들여쓰기를 위해 내용을 줄바꿈 처리
        for line in content_preview.split('\n'):
            print(f"      {line}")


# ==========================================
# 대화형 챗봇 인터페이스
# ==========================================
def run_general_openai_chatbot(client):
    """일반 OpenAI 대화형 챗봇"""
    print("\n" + "=" * 80)
    print("일반 OpenAI 챗봇 (학습 데이터만 사용)")
    print("=" * 80)
    print("\n명령어:")
    print("  - 질문 입력: 아무거나 물어보세요")
    print("  - 'q' 입력: 종료")
    print("  - 'switch' 입력: RAG 챗봇으로 전환")
    print("\n" + "─" * 80)
    
    while True:
        print("\n")
        user_input = input("질문: ").strip()
        
        if not user_input:
            print("질문을 입력해주세요.")
            continue
        
        if user_input.lower() == 'q':
            print("\n일반 OpenAI 챗봇을 종료합니다.")
            break
        
        if user_input.lower() == 'switch':
            return 'switch'
        
        # 답변 생성
        print("\n[일반 OpenAI 응답]")
        print("─" * 80)
        answer, elapsed_time = general_openai_chat(client, user_input)
        print(answer)
        print(f"\n응답 시간: {elapsed_time:.2f}초")
        print("─" * 80)
    
    return None


def run_rag_chatbot(rag_system):
    """RAG 기반 대화형 챗봇"""
    print("\n" + "=" * 80)
    print("RAG 챗봇 (문서 + AI)")
    print("=" * 80)
    print("\n명령어:")
    print("  - 질문 입력: 문서 기반으로 답변합니다")
    print("  - 'q' 입력: 종료")
    print("  - 'switch' 입력: 일반 OpenAI 챗봇으로 전환")
    print("\n" + "─" * 80)
    
    chain = rag_system["chain"]
    retriever = rag_system["retriever"]
    
    while True:
        print("\n")
        user_input = input("질문: ").strip()
        
        if not user_input:
            print("질문을 입력해주세요.")
            continue
        
        if user_input.lower() == 'q':
            print("\nRAG 챗봇을 종료합니다.")
            break
        
        if user_input.lower() == 'switch':
            return 'switch'
        
        # 검색된 문서 먼저 가져오기
        print("\n문서 검색 중...")
        retrieved_docs = get_retrieved_docs(retriever, user_input)
        
        # 답변 생성
        print("\n[RAG 응답]")
        print("─" * 80)
        answer, elapsed_time = rag_chat(chain, user_input)
        print(answer)
        print(f"\n응답 시간: {elapsed_time:.2f}초")
        
        # 참조 문서 출력
        display_retrieved_docs(retrieved_docs)
        print("\n" + "─" * 80)
    
    return None


def run_comparison_chatbot(client, rag_system):
    """일반 OpenAI와 RAG 비교 챗봇"""
    print("\n" + "=" * 80)
    print("비교 모드 (일반 OpenAI vs RAG 동시 비교)")
    print("=" * 80)
    print("\n명령어:")
    print("  - 질문 입력: 두 방식의 답변을 동시에 비교합니다")
    print("  - 'q' 입력: 종료")
    print("\n" + "─" * 80)
    
    chain = rag_system["chain"]
    retriever = rag_system["retriever"]
    
    while True:
        print("\n")
        user_input = input("질문: ").strip()
        
        if not user_input:
            print("질문을 입력해주세요.")
            continue
        
        if user_input.lower() == 'q':
            print("\n비교 챗봇을 종료합니다.")
            break
        
        # 일반 OpenAI 답변
        print("\n" + "=" * 80)
        print("[일반 OpenAI 응답]")
        print("─" * 80)
        general_answer, general_time = general_openai_chat(client, user_input)
        print(general_answer)
        print(f"\n응답 시간: {general_time:.2f}초")
        
        # RAG 답변
        print("\n" + "=" * 80)
        print("[RAG 응답]")
        print("─" * 80)
        
        # 검색된 문서 가져오기
        retrieved_docs = get_retrieved_docs(retriever, user_input)
        
        rag_answer, rag_time = rag_chat(chain, user_input)
        print(rag_answer)
        print(f"\n응답 시간: {rag_time:.2f}초")
        
        # 참조 문서 출력
        display_retrieved_docs(retrieved_docs)
        
        # 차이 분석
        print("\n" + "=" * 80)
        print("분석")
        print("─" * 80)
        print(f"  • 일반 OpenAI 응답 길이: {len(general_answer)}자")
        print(f"  • RAG 응답 길이: {len(rag_answer)}자")
        time_diff = abs(rag_time - general_time)
        faster = "RAG" if rag_time < general_time else "일반 OpenAI"
        print(f"  • 속도: {faster}가 {time_diff:.2f}초 빠름")
        print("=" * 80)
    
    return None


# ==========================================
# 메인 메뉴
# ==========================================
def show_main_menu():
    """메인 메뉴 출력"""
    print("\n" + "=" * 80)
    print("대화형 챗봇 시스템")
    print("=" * 80)
    print("\n모드를 선택하세요:")
    print("  1. 일반 OpenAI 챗봇 (학습 데이터만)")
    print("  2. RAG 챗봇 (문서 + AI)")
    print("  3. 비교 모드 (두 방식 동시 비교)")
    print("  q. 종료")
    print("\n" + "─" * 80)


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("RAG vs 일반 OpenAI 대화형 챗봇")
    print("=" * 80)
    
    # 1. API 키 확인
    if not check_api_key():
        return
    
    # 2. PDF 파일 확인
    if not check_pdf_exists(CONFIG["PDF_PATH"]):
        return
    
    # 3. 일반 OpenAI 설정
    print("\n[초기화 1/2] 일반 OpenAI 클라이언트 준비 중...")
    client = setup_general_openai()
    if not client:
        return
    print("✓ 일반 OpenAI 준비 완료")
    
    # 4. RAG 시스템 설정
    print("\n[초기화 2/2] RAG 시스템 준비 중...")
    rag_system = setup_rag_system(CONFIG["PDF_PATH"])
    if not rag_system:
        return
    
    # 5. 메인 루프
    while True:
        show_main_menu()
        choice = input("\n선택: ").strip()
        
        if choice == 'q' or choice.lower() == 'quit':
            print("\n프로그램을 종료합니다. 감사합니다!")
            break
        
        elif choice == '1':
            result = run_general_openai_chatbot(client)
            if result == 'switch':
                print("\nRAG 챗봇으로 전환합니다...")
                run_rag_chatbot(rag_system)
        
        elif choice == '2':
            result = run_rag_chatbot(rag_system)
            if result == 'switch':
                print("\n일반 OpenAI 챗봇으로 전환합니다...")
                run_general_openai_chatbot(client)
        
        elif choice == '3':
            run_comparison_chatbot(client, rag_system)
        
        else:
            print("\n잘못된 선택입니다. 1, 2, 3, 또는 q를 입력하세요.")


if __name__ == "__main__":
    main()