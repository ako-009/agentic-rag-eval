from typing import TypedDict, List, Optional


class RAGState(TypedDict):
    # Input
    query: str
    reformulated_query: Optional[str]

    # Routing
    source_type: str        # "vectorstore" or "web"
    live_data_used: bool

    # Retrieval
    retrieved_chunks: List[str]

    # Generation
    answer: str

    # Critic
    faithfulness_score: float
    is_hallucinated: bool
    critic_reason: str

    # Control flow
    retry_count: int
    final_answer: str