from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from similarity import cosine_similarity

TOP_K = 3
SIMILARITY_THRESHOLD = 0.30

load_dotenv()


# Initialize the embedding model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


# Step 1: Load and split the document
def load_and_split_documents():
    document_path = Path("documents/python_basics.txt")
    document_text = document_path.read_text(encoding="utf-8")

    document = Document(
        page_content=document_text, metadata={"source": str(document_path)}
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=15)

    return text_splitter.split_documents([document])


# Step 2: Generate embeddings for all chunks
def generate_chunk_embeddings(chunks):
    return embeddings_model.embed_documents([chunk.page_content for chunk in chunks])


# Step 3: Search and rank relevant chunks
def search_relevant_chunks(question, chunk_embeddings):
    question_embedding = embeddings_model.embed_query(question)

    scores = []

    for index, chunk_embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(question_embedding, chunk_embedding)

        if score >= SIMILARITY_THRESHOLD:
            scores.append((score, index))

    scores.sort(reverse=True)

    return scores[:TOP_K]


# Load document and generate embeddings
chunks = load_and_split_documents()
chunk_embeddings = generate_chunk_embeddings(chunks)


# Run the search when this file is executed directly
if __name__ == "__main__":
    question = input("Enter your question: ").strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    top_chunks = search_relevant_chunks(question, chunk_embeddings)

    if not top_chunks:
        print("No relevant information found in the document.")

    else:
        for score, index in top_chunks:
            print(f"\nSimilarity score: {score:.4f}")
            print(f"Chunk index: {index}")
            print(chunks[index].page_content)
