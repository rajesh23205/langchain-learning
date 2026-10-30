from langchain_openai import ChatOpenAI

from search_documents import (
    chunk_embeddings,
    chunks,
    search_relevant_chunks,
)

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
)


if __name__ == "__main__":
    question = input("Enter your question: ").strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    # Find the most relevant document chunks
    top_chunks = search_relevant_chunks(question, chunk_embeddings)

    # Stop if no relevant context was found
    if not top_chunks:
        print("I couldn't find an answer in the provided document.")

    else:
        context = "\n\n".join(chunks[index].page_content for score, index in top_chunks)

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

        for score, index in top_chunks:
            print(f"\nSource: {chunks[index].metadata['source']}")
            print(f"Chunk index: {index}")
            print(f"Similarity score: {score:.4f}")
            print(f"Content: {chunks[index].page_content}")
