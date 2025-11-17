import asyncio
from langchain_community.vectorstores import Redis
from langchain_core.documents import Document
from models.embeddings import get_embeddings
from config.config import Config


class RAGPipeline:

    def __init__(self):
        self.url = Config.REDIS_URL
        self.index_name = Config.REDIS_INDEX_NAME
        self.embedding_function = get_embeddings()
        self.client = None

        try:
            self.client = Redis.from_existing_index(
                redis_url=self.url,
                index_name=self.index_name,
                embedding=self.embedding_function,
                schema="schema.yaml"
            )
        except Exception:
            self.client = Redis(
                redis_url=self.url,
                index_name=self.index_name,
                embedding=self.embedding_function
            )

    def get_retriever(self, k=10):
        return self.client.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    async def add_documents(self, documents):

        docs = []

        for d in documents:
            if isinstance(d, str):
                docs.append(Document(page_content=d))
            else:
                docs.append(d)

        try:
            redis_client = Redis.from_existing_index(
                redis_url=self.url,
                embedding=self.embedding_function,
                index_name=self.index_name,
                schema="schema.yaml"
            )

            redis_client.add_documents(
                documents=docs,
                embedding_function=self.embedding_function,
                index_name=self.index_name,
                redis_url=self.url
            )

        except ValueError:

            redis_client = Redis(
                redis_url=self.url,
                embedding=self.embedding_function,
                index_name=self.index_name
            )

            redis_client.from_documents(
                documents=docs,
                embedding=self.embedding_function,
                index_name=self.index_name,
                redis_url=self.url
            )

    def query(self, text, k=4):
        retriever = self.get_retriever(k=k)
        return retriever.invoke(text)
