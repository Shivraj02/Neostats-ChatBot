import asyncio
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from utils.rag_utils import RAGPipeline
from utils.search_utils import web_search
from models.llm import get_llm
from utils.other_utils import apply_mode, chunk_documents

st.title("Smart Assistant (RAG + Web Search + Gemini)")


mode = st.radio("Response Mode", ["concise", "detailed"])
use_websearch = st.checkbox("🔍 Enable Web Search (Tavily)")
rag = RAGPipeline()

uploaded = st.file_uploader("Upload files", type=["txt", "md", "pdf"])

if uploaded:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    raw_docs = []

    if uploaded.type == "application/pdf":
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # Pages are already Documents → clean text
        for p in pages:
            cleaned = " ".join(p.page_content.split())      # remove weird spaces
            raw_docs.append(Document(page_content=cleaned))

    else:
        loader = TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()

        for d in docs:
            cleaned = " ".join(d.page_content.split())
            raw_docs.append(Document(page_content=cleaned))

    print("Adding documents to RAG..., total:", len(raw_docs), raw_docs)
    # CHUNK THEM PROPERLY
    chunks = chunk_documents(raw_docs)

    asyncio.run(rag.add_documents(chunks))
    st.success("Document added to Redis vector DB!")



user_query = st.text_input("Ask something:")
btn = st.button("Submit")

if btn and user_query:
    llm = get_llm(mode)

    context_docs = rag.query(user_query, k=5)
    rag_context = "\n".join([doc.page_content for doc in context_docs])
    web_context = ""
    if use_websearch:
        st.info("Running web search...")
        web_context = web_search(user_query)

    final_context = rag_context

    if use_websearch:
        final_context += "\n\n---\nWeb Search Results:\n" + web_context

    print("Final Context:", final_context)

    final_prompt = f"Context:\n{final_context}\n\nQuestion: {user_query}"
    final_prompt = apply_mode(final_prompt, mode)

    result = llm.invoke(final_prompt)
    st.write(result.content)
