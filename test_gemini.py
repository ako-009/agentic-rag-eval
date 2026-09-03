import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

def test_gemini_connection():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found")
        return False
    print(f"API key loaded: {api_key[:8]}...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0.1
    )
    print("LLM initialized")
    print("Sending test message...")
    response = llm.invoke([HumanMessage(content="Say exactly: GEMINI CONNECTION SUCCESSFUL")])
    print(f"Response: {response.content}")
    print("Phase 2 Complete!")
    return True

if __name__ == "__main__":
    test_gemini_connection()