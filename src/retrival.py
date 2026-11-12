
import requests
import chromadb

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "chroma_db"

MODEL_NAME = "qwen3-embedding:0.6b"
OLLAMA_URL = "http://localhost:11434/api/embed"

QUERY_INSTRUCTION = (
    "Instruct: Given a question in Hindi or English, "
    "retrieve the most relevant passages from a Hindi document.\n"
    "Query: "
)


def load_retrivals():
    client = chromadb.PersistentClient(path=str(DB_PATH))

    try:
        collection = client.get_collection(name="hindi_rag")
    except Exception as error:
        raise RuntimeError(
            "ChromaDB collection not found. "
            "Run embeddings.py and vector_store.py first."
        ) from error

    return MODEL_NAME, collection


def create_query_embedding(query, model):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "input": QUERY_INSTRUCTION + query,
            "truncate": True
        },
        timeout=300
    )

    response.raise_for_status()

    vector = response.json()["embeddings"][0]

    # Normalize query vector in the same way as document vectors.
    norm = sum(value * value for value in vector) ** 0.5

    if norm == 0:
        raise ValueError("Query embedding has zero magnitude.")

    return [value / norm for value in vector]


def retrival(query, model, collection, top_k=10):
    if not query or not query.strip():
        return []

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    query_vector = create_query_embedding(query, model)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, total_chunks)
    )

    retrieved_chunks = []

    for i in range(len(results["ids"][0])):
        retrieved_chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return retrieved_chunks


def main():
    model, collection = load_retrivals()

    print("Total stored chunks:", collection.count())

    question = input("Enter your question: ")

    results = retrival(
        question,
        model,
        collection,
        top_k=10
    )

    if not results:
        print("No relevant chunks found.")
        return

    for item in results:
        print("\nChunk ID:", item["chunk_id"])
        print("Page:", item["metadata"]["page"])
        print("Distance:", round(item["distance"], 4))
        print("Text:", item["text"])


if __name__ == "__main__":
    main()
