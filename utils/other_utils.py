from langchain_text_splitters import RecursiveCharacterTextSplitter
def apply_mode(prompt, mode):
    if mode == "concise":
        return f"Give a short answer:\n{prompt}"
    return f"Give a very detailed answer:\n{prompt}"


def chunk_documents(raw_documents, chunk_size=800, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
    )

    return splitter.split_documents(raw_documents)
