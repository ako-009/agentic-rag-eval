from langchain_core.messages import HumanMessage
from app.generation.llm import default_llm


REFORMULATE_PROMPT = """You are an expert at query reformulation for document retrieval systems.

The original query failed to retrieve relevant information. Your job is to rewrite it
to be more specific, use different keywords, or break it into a more targeted question.

Original query: {query}
Reason retrieval failed: {reason}
Attempt number: {attempt}

Rules:
- Use different keywords than the original
- Be more specific
- Keep it as a question
- Return ONLY the rewritten query, nothing else

Rewritten query:"""


def reformulate_query(
    original_query: str,
    reason: str,
    attempt: int = 1,
) -> str:
    """
    Rewrites a query that failed to retrieve relevant context.

    This is the key to self-healing — instead of giving up,
    we try a different angle to find the right information.

    Args:
        original_query: The query that led to hallucination
        reason: Why the critic flagged the answer
        attempt: Which retry attempt this is

    Returns:
        A reformulated query string
    """
    prompt = REFORMULATE_PROMPT.format(
        query=original_query,
        reason=reason,
        attempt=attempt,
    )

    response = default_llm.invoke([HumanMessage(content=prompt)])
    reformulated = response.content.strip()

    # Clean up if LLM adds quotes or extra text
    reformulated = reformulated.strip('"').strip("'")
    if "\n" in reformulated:
        reformulated = reformulated.split("\n")[0]

    return reformulated