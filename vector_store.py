from langchain_chroma import Chroma

from search_documents import chunks, embeddings_model

DISTANCE_THRESHOLD = 0.80

# Connect to the existing ChromaDB collection
vector_store = Chroma(
    collection_name="python_basics",
    embedding_function=embeddings_model,
    persist_directory="./chroma_db",
)


# Add chunks only if the collection is empty
if vector_store._collection.count() == 0:
    vector_store.add_documents(documents=chunks)
    print("Documents added to ChromaDB.")

else:
    print("Existing documents found. Skipping insertion.")


print("Total documents:", vector_store._collection.count())


# Search for similar documents
# Search for similar documents
question = "What is the capital of Japan?"

results = vector_store.similarity_search_with_score(
    query=question,
    k=3,
)

print("\nRelevant documents:")

relevant_documents = [
    (document, distance)
    for document, distance in results
    if distance <= DISTANCE_THRESHOLD
]

if not relevant_documents:
    print("No relevant information found in the document.")

else:
    for index, (document, distance) in enumerate(relevant_documents, start=1):
        print(f"\nResult {index}:")
        print(f"Distance score: {distance:.4f}")
        print(document.page_content)
