from app.critic.hallucination import detect_hallucination


def compute_faithfulness_score(
    answer: str,
    retrieved_chunks: list[str],
) -> float:
    """
    Returns a faithfulness score between 0.0 and 1.0.

    1.0 = fully grounded in context
    0.0 = completely hallucinated

    This score is what RAGAS also measures — we compute it
    here using LLM-as-judge before running formal RAGAS eval.
    """
    result = detect_hallucination(answer, retrieved_chunks)
    return result["faithfulness_score"]


def is_answer_faithful(
    answer: str,
    retrieved_chunks: list[str],
    threshold: float = 0.7,
) -> bool:
    """
    Returns True if answer is faithful to context above threshold.
    """
    score = compute_faithfulness_score(answer, retrieved_chunks)
    return score >= threshold