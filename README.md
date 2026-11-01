# LangChain Learning

A hands-on journey into **LangChain, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and AI Agents** using Python.

This repository documents my practical learning journey, with small implementations and experiments to understand how LangChain works internally and how its components fit together.

The focus is on learning concepts step by step through working code.

---

## Learning Objectives

- Understand LangChain fundamentals and its core abstractions.
- Work with prompts, chat models, and LCEL chains.
- Generate embeddings and perform semantic similarity search.
- Build document-based RAG applications.
- Store and retrieve embeddings using ChromaDB.
- Create tools and AI assistants.
- Generate answers grounded in retrieved documents.
- Preserve and display source references in RAG responses.

---

## Features

### 1. LangChain Fundamentals
- Chat model initialization using OpenAI.
- Prompt templates with `ChatPromptTemplate`.
- Structured model output using Pydantic.
- LCEL chain composition.
- `RunnableLambda` and `RunnablePassthrough`.
- Streaming model responses.
- Error handling and logging.

### 2. Document Processing
- Load text documents.
- Convert documents into LangChain `Document` objects.
- Split documents into smaller chunks.
- Preserve source metadata during document processing.

### 3. Embeddings and Vector Search
- Generate embeddings using OpenAI embedding models.
- Store document embeddings in ChromaDB.
- Perform semantic similarity searches.
- Filter retrieved chunks using a distance threshold.

### 4. Retrieval-Augmented Generation (RAG)
- Retrieve relevant document chunks.
- Pass retrieved context to an LLM.
- Generate answers based on document content.
- Handle questions when relevant information is unavailable.

### 5. Tools and AI Assistants
- Create custom tools using LangChain's `@tool` decorator.
- Implement a knowledge-base search tool.
- Implement a calculator tool.
- Build an assistant that retrieves document context before generating an answer.

### 6. Source-Aware RAG
- Preserve source metadata for retrieved chunks.
- Include source filenames in the retrieved context.
- Display source references alongside generated answers.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.13 | Programming language |
| LangChain | LLM application framework |
| OpenAI | Chat models and embeddings |
| ChromaDB | Vector database |
| NumPy | Numerical operations and similarity calculations |
| Pydantic | Structured data validation |
| python-dotenv | Environment variable management |

---

## Project Structure

```text
langchain-learning/
│
├── documents/
│   └── machine_learning.txt
│
├── chroma_db/                 # Local persistent vector database
│
├── .env                       # API keys and model configuration
├── .gitignore
├── requirements.txt
├── README.md
│
├── main.py                    # LangChain fundamentals and LCEL
├── load_document.py           # Document loading
├── generate_embeddings.py     # Document chunking and embeddings
├── similarity.py              # Cosine similarity calculations
├── search_documents.py        # Document loading and splitting
├── vector_store.py            # ChromaDB ingestion and retrieval
├── rag_answer.py              # Basic document-based RAG
├── tools.py                   # Custom LangChain tools
└── agent.py                   # Document-based AI assistant
```

**Note:** `chroma_db/` is generated locally when the vector database is initialized and is excluded from Git.

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/rajesh23205/langchain-learning.git
```

### 2. Navigate to the project directory

```bash
cd langchain-learning
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file in the project root.

Add your OpenAI configuration:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_DEFAULT_MODEL=your_supported_openai_model
```

Replace the placeholders with your own configuration.

**Important:** Never commit your `.env` file or expose your API key publicly.

---

## Running the Project

Run the following commands from the project root with your virtual environment activated.

### 1. LangChain Fundamentals

```bash
python main.py
```

Demonstrates prompt templates, structured output, LCEL, runnable components, and streaming.

### 2. Document Loading

```bash
python load_document.py
```

Loads and splits the documents.

### 3. Embedding Generation

```bash
python generate_embeddings.py
```

Processes document chunks and generates embeddings.

### 4. Vector Store Ingestion

```bash
python vector_store.py
```

Synchronizes document chunks with the persistent ChromaDB collection.

### 5. Basic RAG

```bash
python rag_answer.py
```

Retrieves relevant document chunks and generates an answer using the retrieved context.

### 6. Test Custom Tools

```bash
python tools.py
```

Tests the knowledge-base search tool.

### 7. Document-Based AI Assistant

```bash
python agent.py
```

Retrieves relevant document context before generating an answer.

---

## Learning Progress

| Phase | Topic | Status |
|---|---|---|
| Phase 1 | Foundations | Completed |
| Phase 2 | Prompts and Models | Completed |
| Phase 3 | Chains and LCEL | Completed |
| Phase 4 | Documents and RAG | Completed |
| Phase 5 | Tools and Agents | Completed |
| Phase 6 | Source-Aware RAG | Completed |
| Phase 7 | Upcoming | Not started |
| Phase 8 | Upcoming | Not started |
| Phase 9 | Upcoming | Not started |

---

## Key Learning Approach

This repository is built through hands-on practice.

Each phase introduces new concepts, implements them in Python, tests the results, and builds upon previously completed work.

The objective is to understand not just how to use LangChain, but also how its components work together to build practical AI applications.

---

**Status:** Actively learning and expanding the project.