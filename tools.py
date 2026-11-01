from langchain.tools import tool

from vector_store import DISTANCE_THRESHOLD, vector_store


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the result."""
    return a * b


@tool
def search_knowledge_base(question: str) -> str:
    """Search the knowledge base and return relevant information for a question."""

    results = vector_store.similarity_search_with_score(
        query=question,
        k=5,
    )

    relevant_documents = [
        {
            "content": document.page_content,
            "source": document.metadata.get("source", "Unknown source"),
        }
        for document, distance in results
        if distance <= DISTANCE_THRESHOLD
    ]

    if not relevant_documents:
        return "No relevant information found in the knowledge base."

    return "\n\n".join(
        f"Source: {document['source']}\nContent: {document['content']}"
        for document in relevant_documents
    )


tools = [multiply, search_knowledge_base]

if __name__ == "__main__":
    result = search_knowledge_base.invoke({"question": "What is deep learning?"})

    print(result)
