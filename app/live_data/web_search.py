import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def search_web(query: str, max_results: int = 3) -> list[str]:
    """
    Search the web using Tavily and return clean text chunks.

    Tavily is designed for LLM applications — it returns
    clean, relevant snippets rather than raw HTML.

    Args:
        query: the search query
        max_results: number of results to return

    Returns:
        list of text chunks ready to inject into LLM context
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in .env")

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
    )

    chunks = []
    for result in response.get("results", []):
        title = result.get("title", "")
        content = result.get("content", "")
        url = result.get("url", "")
        chunk = f"Source: {title}\nURL: {url}\n{content}"
        chunks.append(chunk)

    print(f"[WEB SEARCH] Found {len(chunks)} results for: {query}")
    return chunks