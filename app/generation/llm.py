import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm(temperature=0.1):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found")
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=api_key,
        temperature=temperature,
        max_tokens=500,
    )

default_llm = get_llm()