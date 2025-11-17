import streamlit as st

class Config:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    REDIS_INDEX_NAME = "rag_docs"
    REDIS_URL = st.secrets["REDIS_URL"]
