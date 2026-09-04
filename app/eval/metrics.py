import time
from app.retrieval.retriever import retrieve_relevant_chunks
from app.generation.generator import generate_answer
from app.critic.hallucination import detect_hallucination


def evaluate_single(question: str, ground_truth: str) -> dict:
    """
    Run RAG pipeline on one question and compute metrics.

    Returns:
        dict with question, answer, faithfulness_score,
        is_hallucinated, latency_ms
    """
    start = time.time()

    # Retrieve
    chunks = retrieve_relevant_chunks(question, k=3)

    # Generate
    answer = generate_answer(question, chunks)

    # Critic
    critic_result = detect_hallucination(answer, chunks)

    latency_ms = (time.time() - start) * 1000

    return {
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "retrieved_chunks": chunks,
        "faithfulness_score": critic_result["faithfulness_score"],
        "is_hallucinated": critic_result["is_hallucinated"],
        "critic_reason": critic_result["reason"],
        "latency_ms": latency_ms,
    }


def compute_aggregate_metrics(results: list[dict]) -> dict:
    """
    Compute aggregate metrics across all eval results.

    These are the metrics on your CV:
    - hallucination_rate: % of answers that were hallucinated
    - avg_faithfulness: average faithfulness score
    - avg_latency_ms: average response time
    - total_evaluated: number of questions evaluated
    """
    if not results:
        return {}

    total = len(results)
    hallucinated = sum(1 for r in results if r["is_hallucinated"])
    faithfulness_scores = [r["faithfulness_score"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    return {
        "total_evaluated": total,
        "hallucination_rate": hallucinated / total,
        "hallucination_count": hallucinated,
        "avg_faithfulness": sum(faithfulness_scores) / total,
        "min_faithfulness": min(faithfulness_scores),
        "max_faithfulness": max(faithfulness_scores),
        "avg_latency_ms": sum(latencies) / total,
        "p50_latency_ms": sorted(latencies)[total // 2],
    }