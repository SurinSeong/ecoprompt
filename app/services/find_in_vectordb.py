from app.models.load_for_rag import get_vector_store


def find_documents(user_input: str):
    """Qdrant를 통해 관련 정보 찾기""" 

    vector_store = get_vector_store()

    results = vector_store.similarity_search(user_input, k=2)
    try:
        serialized = "\n\n".join(
            (f"Source: {result.metadata}\nContent: {result.page_content}")
            for result in results
        )
        return serialized
    
    except Exception as e:
        print("Error: {e}")
        return None

