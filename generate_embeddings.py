from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Load the document
document_path = Path("documents/python_basics.txt")
document_text = document_path.read_text(encoding="utf-8")

document = Document(page_content=document_text, metadata={"source": str(document_path)})

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=15)

chunks = text_splitter.split_documents([document])

# Create the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Generate one embedding for each chunk
chunk_embeddings = embeddings_model.embed_documents(
    [chunk.page_content for chunk in chunks]
)

# Display results
print("Total chunks:", len(chunks))
print("Total embeddings:", len(chunk_embeddings))

for index, embedding in enumerate(chunk_embeddings, start=1):
    print(f"Chunk {index}: {len(embedding)} dimensions")
