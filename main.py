import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_DEFAULT_MODEL")


# Define the expected structure of the model's answer
class TopicExplanation(BaseModel):
    topic: str
    definition: str
    purpose: str


# Initialize the LangChain chat model
llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    reasoning_effort="low",
)

print("Model name:", llm.model_name)
print("Model kwargs:", llm.model_kwargs)
print("LangChain model initialized:", model_name)


# Create a reusable prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI tutor."),
        (
            "human",
            "Explain {topic}. Provide its definition and its main purpose.",
        ),
    ]
)


# Get input from the user
topic = input("Enter a topic: ")


# Prepare the prompt
prompt_value = prompt_template.invoke({"topic": topic})

print("\nPrompt type:", type(prompt_value))
print("Prepared prompt:", prompt_value)


# Configure the model to return structured data
structured_llm = llm.with_structured_output(TopicExplanation)


# Send the prepared prompt to the model
structured_response = structured_llm.invoke(prompt_value)


# Inspect the structured response
print("\nResponse type:", type(structured_response))
print("Response data:", structured_response.model_dump())

print("\nTopic:", structured_response.topic)
print("Definition:", structured_response.definition)
print("Purpose:", structured_response.purpose)
