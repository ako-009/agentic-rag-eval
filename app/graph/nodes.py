import json
from langchain_core.messages import HumanMessage
from app.graph.state import RAGState
from app.retrieval.retriever import retrieve_relevant_chunks
from app.generation.generator import generate_answer
from app.generation.llm import default_llm
from app.generation.prompts import CRITIC_PROMPT_TEMPLATE, REFORMULATE_PROMPT_TEMPLATE


def retrieve_node(state: RAGState) -> dict:
    """
    Node 1: Retrieve relevant document chunks.
    
    Uses reformulated_query if available (after a self-heal attempt),
    otherwise uses the original query.
    """
    query = state.get("reformulated_query") or state["query"]
    print(f"[RETRIEVE] Query: {query}")

    chunks = retrieve_relevant_chunks(query, k=3)
    print(f"[RETRIEVE] Found {len(chunks)} chunks")

    return {"retrieved_chunks": chunks}


def generate_node(state: RAGState) -> dict:
    """
    Node 2: Generate answer using LLM + retrieved chunks.
    """
    query = state.get("reformulated_query") or state["query"]
    chunks = state["retrieved_chunks"]

    print(f"[GENERATE] Generating answer for: {query}")
    answer = generate_answer(query, chunks)
    print(f"[GENERATE] Answer: {answer[:100]}...")

    return {"answer": answer}


def critic_node(state: RAGState) -> dict:
    """
    Node 3: Critic checks if answer is grounded in retrieved chunks.
    
    This is the hallucination detection step.
    LLM-as-judge: we ask the LLM to evaluate its own output.
    """
    chunks = state["retrieved_chunks"]
    answer = state["answer"]
    context = "\n\n".join(chunks)

    print(f"[CRITIC] Checking answer grounding...")

    prompt = CRITIC_PROMPT_TEMPLATE.format(
        context=context,
        answer=answer,
    )

    response = default_llm.invoke([HumanMessage(content=prompt)])

    try:
        # Parse JSON response from critic
        raw = response.content.strip()
        # Remove markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())

        faithfulness_score = float(result.get("score", 0.5))
        is_hallucinated = not result.get("is_grounded", True)
        reason = result.get("reason", "No reason provided")

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[CRITIC] Failed to parse response: {e}")
        # Default to not hallucinated if parsing fails
        faithfulness_score = 0.5
        is_hallucinated = False
        reason = "Could not parse critic response"

    print(f"[CRITIC] Score: {faithfulness_score:.2f} | Hallucinated: {is_hallucinated}")
    print(f"[CRITIC] Reason: {reason}")

    return {
        "faithfulness_score": faithfulness_score,
        "is_hallucinated": is_hallucinated,
        "critic_reason": reason,
    }


def reformulate_node(state: RAGState) -> dict:
    """
    Node 4: Reformulate the query when hallucination is detected.
    
    This is the self-healing step — we rewrite the query to
    try to retrieve better, more relevant chunks.
    """
    original_query = state["query"]
    reason = state.get("critic_reason", "Retrieval was insufficient")
    retry_count = state.get("retry_count", 0)

    print(f"[REFORMULATE] Attempt {retry_count + 1} - rewriting query...")

    prompt = REFORMULATE_PROMPT_TEMPLATE.format(
        query=original_query,
        reason=reason,
    )

    response = default_llm.invoke([HumanMessage(content=prompt)])
    reformulated = response.content.strip()

    print(f"[REFORMULATE] New query: {reformulated}")

    return {
        "reformulated_query": reformulated,
        "retry_count": retry_count + 1,
    }


def fallback_node(state: RAGState) -> dict:
    """
    Node 5: Fallback when max retries exceeded.
    
    Instead of hallucinating, we gracefully admit we don't have
    enough information. This is the 'grounded validation' in our CV.
    """
    print("[FALLBACK] Max retries exceeded - returning graceful fallback")

    return {
        "final_answer": "I don't have enough information in my knowledge base to answer this question accurately. Please consult additional sources.",
        "is_hallucinated": False,
    }


def finalize_node(state: RAGState) -> dict:
    """
    Node 6: Set the final answer when critic approves.
    """
    print("[FINALIZE] Answer approved by critic")
    return {"final_answer": state["answer"]}