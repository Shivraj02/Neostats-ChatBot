from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.config import Config

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=Config.GOOGLE_API_KEY
    )
