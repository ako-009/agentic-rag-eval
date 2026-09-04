from langchain_core.messages import HumanMessage
from app.graph.state import RAGState
from app.retrieval.retriever import retrieve_relevant_chunks
from app.generation.generator import generate_answer
from app.critic.hallucination import detect_hallucination
from app.critic.reformulator import reformulate_query


def retrieve_node(state: RAGState) -> dict:
    """
    Node 1: Retrieve relevant document chunks.
    Uses reformulated_query if available, otherwise original query.
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

    print(f"[GENERATE] Generating answer...")
    answer = generate_answer(query, chunks)
    print(f"[GENERATE] Answer: {answer[:100]}...")

    return {"answer": answer}


def critic_node(state: RAGState) -> dict:
    """
    Node 3: Critic checks if answer is grounded in retrieved chunks.
    Now uses our dedicated hallucination detection module.
    """
    chunks = state["retrieved_chunks"]
    answer = state["answer"]

    print(f"[CRITIC] Checking answer grounding...")

    result = detect_hallucination(answer, chunks)

    print(f"[CRITIC] Score: {result['faithfulness_score']:.2f} | Hallucinated: {result['is_hallucinated']}")
    print(f"[CRITIC] Reason: {result['reason']}")

    return {
        "faithfulness_score": result["faithfulness_score"],
        "is_hallucinated": result["is_hallucinated"],
        "critic_reason": result["reason"],
    }


def reformulate_node(state: RAGState) -> dict:
    """
    Node 4: Reformulate the query when hallucination is detected.
    Now uses our dedicated reformulator module.
    """
    original_query = state["query"]
    reason = state.get("critic_reason", "Retrieval was insufficient")
    retry_count = state.get("retry_count", 0)

    print(f"[REFORMULATE] Attempt {retry_count + 1} - rewriting query...")

    reformulated = reformulate_query(
        original_query=original_query,
        reason=reason,
        attempt=retry_count + 1,
    )

    print(f"[REFORMULATE] New query: {reformulated}")

    return {
        "reformulated_query": reformulated,
        "retry_count": retry_count + 1,
    }


def fallback_node(state: RAGState) -> dict:
    """
    Node 5: Graceful fallback when max retries exceeded.
    """
    print("[FALLBACK] Max retries exceeded - returning graceful fallback")

    return {
        "final_answer": "I don't have enough information in my knowledge base to answer this question accurately.",
        "is_hallucinated": False,
    }


def finalize_node(state: RAGState) -> dict:
    """
    Node 6: Set the final answer when critic approves.
    """
    print("[FINALIZE] Answer approved by critic")
    return {"final_answer": state["answer"]}