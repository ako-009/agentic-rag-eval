import json
from langchain_core.messages import HumanMessage
from app.generation.llm import default_llm


HALLUCINATION_PROMPT = """You are an expert fact-checker evaluating whether an AI answer is grounded in the provided context.

Your job:
1. Read the context carefully
2. Read the answer carefully  
3. Check if EVERY claim in the answer is directly supported by the context
4. If the answer says "I don't have enough information" or similar, mark it as grounded (score: 1.0)

Context:
{context}

Answer to evaluate:
{answer}

Respond ONLY with valid JSON, no other text:
{{"is_grounded": true or false, "score": 0.0 to 1.0, "reason": "one sentence explanation"}}

Rules:
- score 0.9-1.0: answer is fully supported by context
- score 0.5-0.9: answer is mostly supported, minor gaps
- score 0.0-0.5: answer contains claims not in context (hallucination)
- is_grounded = true only when score >= 0.7"""


def detect_hallucination(
    answer: str,
    retrieved_chunks: list[str],
    threshold: float = 0.7
) -> dict:
    """
    Detects if an answer is hallucinated relative to retrieved context.

    Args:
        answer: The generated answer to check
        retrieved_chunks: The context the answer should be based on
        threshold: Score below this = hallucinated (default 0.7)

    Returns:
        dict with keys: is_hallucinated, faithfulness_score, reason
    """
    if not retrieved_chunks:
        return {
            "is_hallucinated": True,
            "faithfulness_score": 0.0,
            "reason": "No context was retrieved",
        }

    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = HALLUCINATION_PROMPT.format(
        context=context,
        answer=answer,
    )

    response = default_llm.invoke([HumanMessage(content=prompt)])

    try:
        raw = response.content.strip()
        # Remove thinking tags from Qwen models
        if "<think>" in raw:
            if "</think>" in raw:
                raw = raw.split("</think>")[-1].strip()
            else:
                raw = raw.split("<think>")[0].strip()
        # Remove markdown code blocks
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        score = float(result.get("score", 0.5))
        is_grounded = result.get("is_grounded", score >= threshold)
        reason = result.get("reason", "No reason provided")

        return {
            "is_hallucinated": not is_grounded,
            "faithfulness_score": score,
            "reason": reason,
        }

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[HALLUCINATION] Parse error: {e}, raw: {response.content[:200]}")
        return {
            "is_hallucinated": False,
            "faithfulness_score": 0.5,
            "reason": "Could not parse critic response — defaulting to grounded",
        }