from app.graph.state import RAGState

MAX_RETRIES = 2


def should_retry(state: RAGState) -> str:
    """
    Conditional edge after critic_node.
    
    Decides what happens next based on critic verdict:
    
    1. Not hallucinated → "finalize" (return answer)
    2. Hallucinated + retries left → "reformulate" (self-heal)
    3. Hallucinated + no retries → "fallback" (graceful failure)
    
    This function's return value must match node names in the graph.
    """
    is_hallucinated = state.get("is_hallucinated", False)
    retry_count = state.get("retry_count", 0)

    if not is_hallucinated:
        print(f"[EDGE] Answer is grounded → FINALIZE")
        return "finalize"

    if retry_count < MAX_RETRIES:
        print(f"[EDGE] Hallucination detected, retry {retry_count + 1}/{MAX_RETRIES} → REFORMULATE")
        return "reformulate"

    print(f"[EDGE] Max retries ({MAX_RETRIES}) exceeded → FALLBACK")
    return "fallback"