import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.retrieval.embeddings import default_embeddings

CHROMA_DB_PATH = "chroma_db"

def get_vectorstore():
    """Load existing ChromaDB or create empty one."""
    vectorstore = Chroma(
        collection_name="rag_documents",
        embedding_function=default_embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
    return vectorstore


def ingest_documents(documents: list[str], metadatas: list[dict] = None):
    """
    Split documents into chunks and store in ChromaDB.
    
    documents: list of raw text strings
    metadatas: optional list of dicts with source info
    
    Chunking strategy:
    - chunk_size=500: each chunk is ~500 characters
    - chunk_overlap=50: chunks overlap by 50 chars to avoid cutting mid-sentence
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.create_documents(
        texts=documents,
        metadatas=metadatas or [{}] * len(documents),
    )

    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    vectorstore = Chroma(
        collection_name="rag_documents",
        embedding_function=default_embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    vectorstore.add_documents(chunks)
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore


def get_retriever(k: int = 3):
    """
    Returns a retriever that fetches top-k similar chunks.
    k=3 means: return the 3 most relevant document chunks.
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})