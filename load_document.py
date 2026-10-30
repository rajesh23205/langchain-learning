from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

document_path = Path("documents/python_basics.txt")
document_text = document_path.read_text(encoding="utf-8")

document = Document(page_content=document_text, metadata={"source": str(document_path)})

text_splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=15)

chunks = text_splitter.split_documents([document])

print("Total chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}:")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)
    print("Chunk length:", len(chunk.page_content))
    print("Characters:", repr(chunk.page_content))
