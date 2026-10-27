import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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
)

print("Model name:", llm.model_name)
print("LangChain model initialized:", model_name)


# Create a reusable prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI tutor. Adapt your explanation to the audience specified in the user's request. "
            "Use simple language for beginners and appropriate technical depth for advanced learners. "
            "Explain technical terms when needed and use practical examples when helpful.",
        ),
        (
            "human",
            "Explain {topic} for {audience}. "
            "Include a definition, its main purpose, one practical example, "
            "and exactly three interview questions.",
        ),
    ]
)


# Configure the model to return structured data
structured_llm = llm.with_structured_output(TopicExplanation)


# NEW: Connect the prompt and structured model into a chain
topic_explanation_chain = prompt_template | structured_llm


# Get input from the user
topic = input("Enter a topic: ")
audience = input("Who is the explanation for? ")


# Run the complete chain
structured_response = topic_explanation_chain.invoke(
    {
        "topic": topic,
        "audience": audience,
    }
)


# Display the structured response
print("\nResponse type:", type(structured_response))
print("Response data:", structured_response.model_dump())

print("\nTopic:", structured_response.topic)
print("\nDefinition:", structured_response.definition)
print("\nPurpose:", structured_response.purpose)
print("\nPractical Example:", structured_response.practical_example)

print("\nInterview Questions:")
for index, question in enumerate(
    structured_response.interview_questions,
    start=1,
):
    print(f"{index}. {question}")
