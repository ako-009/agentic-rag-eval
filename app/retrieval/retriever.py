from app.retrieval.vectorstore import get_retriever, ingest_documents
from app.retrieval.embeddings import default_embeddings


def load_documents_from_files(file_paths: list[str]) -> list[str]:
    """Read text files and return their contents as a list of strings."""
    documents = []
    metadatas = []

    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append(content)
        metadatas.append({"source": path})
        print(f"Loaded: {path} ({len(content)} chars)")

    return documents, metadatas


def retrieve_relevant_chunks(query: str, k: int = 3) -> list[str]:
    """
    Given a query string, return top-k most relevant document chunks.
    
    This is the core retrieval step in RAG.
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)

    chunks = [doc.page_content for doc in docs]
    return chunks