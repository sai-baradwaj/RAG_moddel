import re
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:8b"


def generate_answer(question, retrieved_chunks):

    context = ""

    for chunk in retrieved_chunks:

        context += f"""
            [CHUNK_ID: {chunk['chunk_id']}]
            [PAGE: {chunk['metadata']['page']}]

            {chunk['text']}
            """

    prompt = f"""
        You are a strict document-based question-answering assistant.

        Answer the question using ONLY the supplied document context.

        STRICT RULES:

        1. Use only facts explicitly supported by the document context.
        2. Never use pretrained knowledge or external information.
        3. Never invent names, dates, events, explanations or titles.
        4. If the answer is not supported by the context, respond exactly:
        "The provided document does not contain sufficient information."
        5. Answer in the same language as the question.
        6. Keep the answer concise and factual.
        7. Do not generate page numbers or chunk IDs.
        8. Preserve names, abbreviations and terminology exactly as written.
        9. Do not expand abbreviations unless their full form appears in the context.
        10. Do not combine unrelated facts to construct an answer.
        11. If multiple chunks contain relevant evidence, use only the evidence needed.
        12. Do not treat general background information as proof of a specific event.
        13. If the context is ambiguous or contradictory, state that the document
            does not provide sufficient information.
        14. Never add information based on your pretrained knowledge.
        15. If the document says someone died while delivering a lecture,
            do not infer the cause of death.
        16. Answer only the parts of the question supported by the context.
        17. If a question asks how, where, and when, provide only the
            details explicitly available in the retrieved context.
        18. Never introduce institutions, locations, or events that
            are not explicitly mentioned in the context.

        DOCUMENT CONTEXT:

        {context}

        QUESTION:

        {question}

        ANSWER:
        """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=360
    )

    response.raise_for_status()

    result = response.json()
    answer = result["message"]["content"]

    answer = re.sub(
        r'\[CHUNK_ID:\s*chunk_\d+\]',
        '',
        answer
    )

    answer = re.sub(
        r'Chunk[_ ]ID:\s*chunk_\d+',
        '',
        answer,
        flags=re.IGNORECASE
    )

    citations = []

    for chunk in retrieved_chunks:
        citations.append(
            f"Page: {chunk['metadata']['page']}, "
            f"Chunk ID: {chunk['chunk_id']}"
        )

    citation_text = "\n".join(citations)

    return f"""
    {answer}

    Sources:
    {citation_text}
    """
