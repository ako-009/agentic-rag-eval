from langchain_core.messages import HumanMessage
from app.generation.llm import default_llm
from app.generation.prompts import RAG_PROMPT_TEMPLATE


def generate_answer(question: str, retrieved_chunks: list[str]) -> str:
    """
    Generate an answer using the LLM grounded in retrieved chunks.
    
    This is the 'G' in RAG — Retrieval Augmented GENERATION.
    The LLM only uses the provided context, not its training memory.
    """
    context = "\n\n".join(retrieved_chunks)

    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    response = default_llm.invoke([HumanMessage(content=prompt)])
    return response.content