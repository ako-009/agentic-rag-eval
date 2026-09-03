RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based ONLY on the provided context.

If the answer is not in the context, say "I don't have enough information to answer this question."
Do NOT use any knowledge outside of the provided context.

Context:
{context}

Question: {question}

Answer:"""


CRITIC_PROMPT_TEMPLATE = """You are an expert fact-checker. Your job is to verify if an answer is fully supported by the provided context.

Context:
{context}

Answer to verify:
{answer}

Is every claim in the answer directly supported by the context above?
Respond ONLY with valid JSON in this exact format:
{{"is_grounded": true or false, "score": 0.0 to 1.0, "reason": "brief explanation"}}"""


REFORMULATE_PROMPT_TEMPLATE = """The following query did not retrieve relevant information. 
Rewrite it to be more specific and likely to find relevant documents.

Original query: {query}
Reason retrieval failed: {reason}

Provide ONLY the rewritten query, nothing else."""