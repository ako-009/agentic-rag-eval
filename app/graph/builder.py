from langgraph.graph import StateGraph, END
from app.graph.state import RAGState
from app.graph.nodes import (
    retrieve_node,
    generate_node,
    critic_node,
    reformulate_node,
    fallback_node,
    finalize_node,
)
from app.graph.edges import should_retry


def build_rag_graph():
    """
    Builds and compiles the self-healing RAG graph.

    Graph structure:
        START → retrieve → generate → critic → (conditional)
                                                  → finalize → END
                                                  → reformulate → retrieve (loop)
                                                  → fallback → END
    """
    # Initialize graph with our state schema
    graph = StateGraph(RAGState)

    # Add all nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("finalize", finalize_node)

    # Define linear edges (always go here next)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critic")

    # Conditional edge after critic — this is the self-healing loop
    graph.add_conditional_edges(
        "critic",           # From this node
        should_retry,       # Call this function to decide
        {
            "finalize": "finalize",       # If returns "finalize" → go to finalize
            "reformulate": "reformulate", # If returns "reformulate" → go to reformulate
            "fallback": "fallback",       # If returns "fallback" → go to fallback
        }
    )

    # After reformulate → loop back to retrieve (self-healing loop!)
    graph.add_edge("reformulate", "retrieve")

    # Terminal edges
    graph.add_edge("finalize", END)
    graph.add_edge("fallback", END)

    # Compile the graph
    compiled = graph.compile()
    print("Graph compiled successfully")
    return compiled


# Module-level compiled graph
rag_graph = build_rag_graph()