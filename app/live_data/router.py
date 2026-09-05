LIVE_KEYWORDS = [
    "today", "current", "currently", "latest", "now",
    "recent", "recently", "price", "news", "live",
    "2026", "2025", "this week", "this month",
    "right now", "at the moment", "real-time",
    "stock", "weather", "score", "result"
]


def should_use_web_search(query: str) -> bool:
    """
    Decides whether a query needs live web data or ChromaDB.

    Live keywords signal that the user wants real-time info
    that a static knowledge base cannot provide.

    Args:
        query: the user's question

    Returns:
        True → use Tavily web search
        False → use ChromaDB (default)
    """
    query_lower = query.lower()
    for keyword in LIVE_KEYWORDS:
        if keyword in query_lower:
            print(f"[ROUTER] Live keyword '{keyword}' detected → WEB SEARCH")
            return True

    print(f"[ROUTER] No live keywords → VECTORSTORE")
    return False


def get_source_type(query: str) -> str:
    """Returns 'web' or 'vectorstore' for the given query."""
    return "web" if should_use_web_search(query) else "vectorstore"