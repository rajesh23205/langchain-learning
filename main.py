import logging
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s: %(message)s",
)

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_DEFAULT_MODEL")


# Define the expected structure of the model's answer
class TopicExplanation(BaseModel):
    topic: str
    definition: str
    purpose: str
    practical_example: str
    interview_questions: list[str] = Field(min_length=3, max_length=3)


# Initialize the LangChain chat model
llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    reasoning_effort="low",
    timeout=30,
    max_retries=2,
)

streaming_llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    reasoning_effort="low",
    timeout=30,
    max_retries=2,
    streaming=True,
)

print("Model name:", llm.model_name)
print("LangChain model initialized:", model_name)


# Create a reusable prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful AI tutor. Adapt your explanation to the audience specified in the user's request. "
                "Use simple language for beginners and appropriate technical depth for advanced learners. "
                "Explain technical terms when needed and use practical examples when helpful."
            ),
        ),
        (
            "human",
            (
                "Explain {topic} for {audience}. "
                "Include a definition, its main purpose, one practical example, "
                "and exactly three interview questions."
            ),
        ),
    ]
)


# Configure the model to return structured data
structured_llm = llm.with_structured_output(TopicExplanation)


# NEW: Connect the prompt and structured model into a chain
topic_explanation_chain = (prompt_template | structured_llm).with_config(
    run_name="topic_explanation_chain"
)

streaming_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a technical AI tutor. Interpret AI and software "
                "terms in their technical context. In particular, RAG means "
                "Retrieval-Augmented Generation unless the user clearly "
                "indicates another meaning. Keep answers under 150 words."
            ),
        ),
        ("human", "{question}"),
    ]
)

streaming_chain = streaming_prompt | streaming_llm

question_chain = (
    RunnablePassthrough.assign(
        question=RunnableLambda(
            lambda data: f"Explain {data['topic']} for a {data['audience']}."
        )
    )
    | streaming_chain
)

uppercase_chain = RunnableLambda(lambda data: data["topic"].upper())

result = uppercase_chain.invoke({"topic": "RAG"})

print("\nRunnableLambda result:", result)


def display_explanation(response: TopicExplanation):
    print("\nTopic:", response.topic)
    print("\nDefinition:", response.definition)
    print("\nPurpose:", response.purpose)
    print("\nPractical Example:", response.practical_example)

    print("\nInterview Questions:")
    for index, question in enumerate(
        response.interview_questions,
        start=1,
    ):
        print(f"{index}. {question}")


# Get input from the user
topic = input("Enter a topic: ").strip()
audience = input("Who is the explanation for? ").strip()

# Validate the input
if not topic or not audience:
    print("Error: Both topic and audience are required.")
    raise SystemExit(1)


# Run the complete chain
try:
    structured_response = topic_explanation_chain.invoke(
        {
            "topic": topic,
            "audience": audience,
        }
    )
except Exception:
    logging.exception("Error generating explanation")
    raise SystemExit(1)


# Display the structured response
print("\nResponse type:", type(structured_response))
display_explanation(structured_response)

streaming_question = input("\nEnter a question for streaming: ").strip()

if streaming_question:
    print("\nStreaming response:")

    try:
        for chunk in streaming_chain.stream({"question": streaming_question}):
            if chunk.content:
                print(chunk.content, end="", flush=True)

        print()

    except Exception:
        logging.exception("Error generating streaming response")
        raise SystemExit(1)
else:
    print("No streaming question provided.")


passthrough_chain = RunnablePassthrough.assign(
    question=RunnableLambda(
        lambda data: f"Explain {data['topic']} for a {data['audience']}."
    )
)

test_input = {"topic": "RAG", "audience": "Developer"}

result = passthrough_chain.invoke(test_input)

print("\nPassthrough result:", result)

print("\nTesting question_chain:")

for chunk in question_chain.stream({"topic": "RAG", "audience": "Developer"}):
    if chunk.content:
        print(chunk.content, end="", flush=True)

print()
