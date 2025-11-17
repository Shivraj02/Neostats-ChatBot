from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import Config

def get_llm(mode="concise"):
    style = (
        "Provide short and crisp answers." if mode == "concise"
        else "Provide long, detailed, expanded responses."
    )

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        google_api_key=Config.GOOGLE_API_KEY,
        system_instruction=style
    )
