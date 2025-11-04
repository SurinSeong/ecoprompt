from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from app.models.prompt_template import rag_prompt

load_dotenv()


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
        x: {"question": str, "history": str, "context": str, "personal_prompt": str}
        """
        # 방어적 캐스팅
        question = x.get("question", "")
        history = x.get("history", "")
        context = x.get("context", "")
        personal_prompt = x.get("personal_prompt", "")

        if not isinstance(question, str): question = str(question)
        if not isinstance(history, str): history = str(history)
        if not isinstance(context, str): context = str(context)
        if not isinstance(personal_prompt, str): personal_prompt = str(personal_prompt)

        system_prompt = (
            rag_prompt + 
            "\n---\n[사용자 지침]\n" + personal_prompt +
            "\n\n[History]\n" + history +
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
        personal_prompt=RunnablePassthrough(),
        history=RunnablePassthrough(),
        # context=RunnableLambda(lambda _: ""),    # retriever | format_docs 로 교체해야 함.
        context= retriever | format_docs
    )

    chain = Inputs | make_prompt | llm | StrOutputParser()

    return chain