from typing import TypedDict, List, Optional


class RAGState(TypedDict):
    """
    Shared state passed between all nodes in the LangGraph.
    
    Every node receives this state, does its job, and returns
    an updated version. LangGraph merges the updates automatically.
    
    Think of this as the 'context' of one complete RAG request.
    """

    # Input
    query: str                          # Original user question

    # Retrieval
    reformulated_query: Optional[str]   # Rewritten query if retrieval failed
    retrieved_chunks: List[str]         # Top-k document chunks from ChromaDB

    # Generation
    answer: str                         # LLM's generated answer

    # Critic
    faithfulness_score: float           # How grounded is the answer? (0.0-1.0)
    is_hallucinated: bool               # Did the critic detect hallucination?
    critic_reason: str                  # Why the critic flagged it

    # Control flow
    retry_count: int                    # How many self-heal attempts so far
    final_answer: str                   # The answer we return to the user