from tavily import TavilyClient
from config.config import Config

client = TavilyClient(api_key=Config.TAVILY_API_KEY)

def web_search(query):
    results = client.search(query=query, max_results=4)
    combined = "\n".join([r["content"] for r in results["results"]])
    return combined
