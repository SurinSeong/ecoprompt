import json
import httpx
from typing import AsyncGenerator, Dict, Any

from vllm.v1.engine.async_llm import AsyncLLM
from vllm import SamplingParams

from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from app.models.prompt_template import rag_prompt

load_dotenv()


# # 모델에게 답변 받기 - vllm
async def generate_sse_stream(
        llm_engine: AsyncLLM,
        request_id: str,
        user_input: str,
        sampling_params: SamplingParams,
        final_responses: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """
    n=2 응답 처리:
        outputs[0]은 스트리밍, outputs[1]은 컨테이너에 저장.
        (sampling_params의 n은 반드시 2로 설정되어 있어야 함.)
    """
    agen = llm_engine.generate(
        request_id=request_id, prompt=user_input, sampling_params=sampling_params
    )

    sent_text = ""
    rejected_text = ""    # 두 번째 응답의 누적 텍스트를 저장할 변수

    try:
        async for result in agen:
            if not result.outputs or len(result.outputs) < 2:
                continue

            # Chosen: 스트리밍 처리
            chosen_text = result.outputs[0].text
            new_text = chosen_text[len(sent_text):]
            sent_text = chosen_text

            if new_text:
                # SSE 및 yield
                payload = {"delta": new_text}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # rejected: 누적 처리 후 저장
            rejected_text = result.outputs[1].text

            # 최종 결과 저장 및 종료
            if result.finished:
                # 스트리밍 루프 끝난 후, 최종 텍스트를 컨테이너에 저장
                final_responses["chosen"] = chosen_text
                final_responses["rejected"] = rejected_text

                # 스트리밍 완료
                yield "event: end_of_stream\ndata: {}\n\n"
                return
            
    except Exception as e:
        error_payload = {"error": f"Error during streaming: {str(e)}"}
        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
        raise 


# RAG chain을 통해 답변 생성하기
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def stream_response(vector_store, llm, tokenizer):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 1
        }
    )

    def build_prompt_with_template(x: dict) -> str:
        """
        x: {"question": str, "history": str, "context": str}
        """
        # 방어적 캐스팅
        question = x.get("question", "")
        history = x.get("history", "")
        context = x.get("context", "")

        if not isinstance(question, str): question = str(question)
        if not isinstance(history, str): history = str(history)
        if not isinstance(context, str): context = str(context)

        system_prompt = (
            rag_prompt + 
            "\n[History]\n" + history +
            "\n\n[Context]\n" + context +
            "\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    make_prompt = RunnableLambda(build_prompt_with_template)

    llm_pipeline = pipeline(
        model=llm,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.3,
        repetition_penalty=1.01,
        return_full_text=False,
        max_new_tokens=1024,
    )

    llm = HuggingFacePipeline(
        pipeline=llm_pipeline
    )

    Inputs = RunnableParallel(
        question=RunnablePassthrough(),
        history=RunnableLambda(lambda _: ""),
        context=RunnableLambda(lambda _: ""),    # retriever | format_docs 로 교체 가능
    )

    chain = Inputs | make_prompt | llm | StrOutputParser()

    return chain