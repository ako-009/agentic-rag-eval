from langchain_core.messages import HumanMessage
from app.graph.state import RAGState
from app.retrieval.retriever import retrieve_relevant_chunks
from app.generation.generator import generate_answer
from app.critic.hallucination import detect_hallucination
from app.critic.reformulator import reformulate_query
from app.live_data.router import should_use_web_search
from app.live_data.web_search import search_web


def route_node(state: RAGState) -> dict:
    """
    Node 0: Decides whether to use ChromaDB or web search.
    """
    query = state["query"]
    use_web = should_use_web_search(query)
    source_type = "web" if use_web else "vectorstore"
    return {
        "source_type": source_type,
        "live_data_used": use_web,
    }


def retrieve_node(state: RAGState) -> dict:
    """
    Node 1: Retrieve from ChromaDB OR web based on routing decision.
    """
    query = state.get("reformulated_query") or state["query"]
    source_type = state.get("source_type", "vectorstore")

    if source_type == "web":
        print(f"[RETRIEVE] Web search for: {query}")
        chunks = search_web(query, max_results=3)
    else:
        print(f"[RETRIEVE] ChromaDB search for: {query}")
        chunks = retrieve_relevant_chunks(query, k=3)

    print(f"[RETRIEVE] Found {len(chunks)} chunks from {source_type}")
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
        "final_answer": "I don't have enough information to answer this question accurately.",
        "is_hallucinated": False,
    }


def finalize_node(state: RAGState) -> dict:
    """
    Node 6: Set the final answer when critic approves.
    """
    source = state.get("source_type", "vectorstore")
    print(f"[FINALIZE] Answer approved by critic (source: {source})")
    return {"final_answer": state["answer"]}