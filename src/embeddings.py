
import json
import requests
import numpy as np

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "results" / "chunks.json"
EMBEDDINGS_PATH = BASE_DIR / "results" / "embeddings.npy"
METADATA_PATH = BASE_DIR / "results" / "embeddings_metadata.json"

MODEL_NAME = "qwen3-embedding:0.6b"
OLLAMA_URL = "http://localhost:11434/api/embed"
BATCH_SIZE = 16


def load_chunks():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "chunks.json not found. Run chunking.py first."
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No chunks found in chunks.json")

    print(f"Total chunks loaded: {len(chunks)}")
    return chunks


def normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return vectors / norms


def generate_embeddings(chunks):
    print("\nConnecting to Ollama embedding model...")

    all_vectors = []

    texts = [chunk["text"] for chunk in chunks]

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "input": batch,
                "truncate": True
            },
            timeout=300
        )

        response.raise_for_status()

        batch_vectors = response.json()["embeddings"]
        all_vectors.extend(batch_vectors)

        print(
            f"Embedded {min(start + len(batch), len(texts))}"
            f"/{len(texts)} chunks"
        )

    vectors = normalize_vectors(all_vectors)

    print("\nEmbedding generation completed!")
    print("Embedding shape:", vectors.shape)

    return vectors


def save_embeddings(chunks, vectors):
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_PATH, vectors)

    metadata = []

    for chunk in chunks:
        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "page": int(chunk["page"]),
            "text": chunk["text"]
        })

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nEmbeddings and metadata saved!")
    print("Embeddings:", EMBEDDINGS_PATH)
    print("Metadata:", METADATA_PATH)


def main():
    chunks = load_chunks()

    vectors = generate_embeddings(chunks)

    save_embeddings(chunks, vectors)

    print("\nEmbedding overview:")
    print("Total chunks:", len(chunks))
    print("Embedding shape:", vectors.shape)
    print("Vector dimensions:", vectors.shape[1])


if __name__ == "__main__":
    main()
