from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tools import search_knowledge_base

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
)


def answer_question(question: str) -> str:

    # Always search the knowledge base first
    context = search_knowledge_base.invoke({"question": question})

    # Stop if no relevant documents are found
    if context == "No relevant information found in the knowledge base.":
        return "I couldn't find information about this question in the available documents."

    # Instruct the LLM to answer only from retrieved documents
    messages = [
        SystemMessage(
            content=(
                "You are a document-based AI assistant. "
                "Answer the user's question using ONLY the provided document context. "
                "Do not use your general knowledge or introduce unsupported facts. "
                "Use the source filename provided with the document context. "
                "At the end of your answer, include a Sources section listing the source filenames. "
                "If the context does not contain enough information, "
                "clearly say that the documents do not provide the answer."
            )
        ),
        HumanMessage(
            content=f"""
            Document context:
            {context}

            User question:
            {question}
            """
        ),
    ]

    response = llm.invoke(messages)

    return response.content


if __name__ == "__main__":
    question = "Explain quantum computing."

    print("Question:", question)
    print("Answer:", answer_question(question))
