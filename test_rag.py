import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.retrieval.vectorstore import ingest_documents
from app.retrieval.retriever import load_documents_from_files, retrieve_relevant_chunks
from app.generation.generator import generate_answer

def test_rag_pipeline():
    print("=" * 50)
    print("Phase 3: Basic RAG Pipeline Test")
    print("=" * 50)

    # Step 1: Ingest documents
    print("\n[1] Ingesting documents...")
    documents, metadatas = load_documents_from_files([
        "data/documents/company_policy.txt"
    ])
    ingest_documents(documents, metadatas)

    # Step 2: Test retrieval + generation
    test_questions = [
        "What is the refund policy?",
        "How many days of paid leave do employees get?",
        "What is the expense approval limit for managers?",
    ]

    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")

        # Retrieve
        chunks = retrieve_relevant_chunks(question, k=3)
        print(f"Retrieved {len(chunks)} chunks")

        # Generate
        answer = generate_answer(question, chunks)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    test_rag_pipeline()