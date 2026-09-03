from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    """
    Returns a HuggingFace embedding model.
    
    all-MiniLM-L6-v2:
    - Free, runs locally (no API calls)
    - 384-dimensional vectors
    - Fast and good enough for our use case
    - Downloads ~90MB on first run
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings

default_embeddings = get_embeddings()