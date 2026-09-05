from langgraph.graph import StateGraph, END
from app.graph.state import RAGState
from app.graph.nodes import (
    route_node,
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
    Builds the self-healing RAG graph with live data routing.

    Graph structure:
        START → route → retrieve → generate → critic → (conditional)
                                                          → finalize → END
                                                          → reformulate → retrieve
                                                          → fallback → END
    """
    graph = StateGraph(RAGState)

    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critic")

    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "finalize": "finalize",
            "reformulate": "reformulate",
            "fallback": "fallback",
        }
    )

    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("finalize", END)
    graph.add_edge("fallback", END)

    compiled = graph.compile()
    print("Graph compiled successfully")
    return compiled


rag_graph = build_rag_graph()