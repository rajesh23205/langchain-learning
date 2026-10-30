from langchain_openai import ChatOpenAI

from vector_store import DISTANCE_THRESHOLD, vector_store

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
)


if __name__ == "__main__":
    question = input("Enter your question: ").strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    # Find the most relevant document chunks
    # Retrieve relevant documents from ChromaDB
    results = vector_store.similarity_search_with_score(
        query=question,
        k=5,
    )

    # Filter documents using the distance threshold
    top_chunks = [
        (document, distance)
        for document, distance in results
        if distance <= DISTANCE_THRESHOLD
    ]

    # Stop if no relevant context was found
    if not top_chunks:
        print("I couldn't find an answer in the provided document.")

    else:
        context = "\n\n".join(
            document.page_content for document, distance in top_chunks
        )

        prompt = f"""
    Answer the question using only the context below.
    If the context does not contain the answer, say you don't know
    based on the provided document. Do not invent information.

    Context:
    {context}

    Question:
    {question}
    """

        response = llm.invoke(prompt)

        print("\nAnswer:", response.content)
        print("\nSources:")

        for index, (document, distance) in enumerate(top_chunks, start=1):
            print(f"\nSource: {document.metadata['source']}")
            print(f"Result number: {index}")
            print(f"Distance score: {distance:.4f}")
            print(f"Content: {document.page_content}")
