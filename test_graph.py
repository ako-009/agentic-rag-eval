import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.graph.builder import rag_graph


def run_query(question: str):
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print('='*60)

    initial_state = {
        "query": question,
        "reformulated_query": None,
        "retrieved_chunks": [],
        "answer": "",
        "faithfulness_score": 0.0,
        "is_hallucinated": False,
        "critic_reason": "",
        "retry_count": 0,
        "final_answer": "",
    }

    result = rag_graph.invoke(initial_state)

    print(f"\nFINAL ANSWER: {result['final_answer']}")
    print(f"Faithfulness Score: {result['faithfulness_score']:.2f}")
    print(f"Hallucinated: {result['is_hallucinated']}")
    print(f"Retries: {result['retry_count']}")
    return result


if __name__ == "__main__":
    # Test 1: Normal question — should answer directly
    run_query("What is the refund policy?")

    # Test 2: Another normal question
    run_query("How many days of paid leave do employees get?")

    # Test 3: Question outside knowledge base — should fallback gracefully
    run_query("What is the stock price of ACME Corporation?")

    # Test 4: Live data question that should get real web answer
    run_query("What is the current Bitcoin price today?")