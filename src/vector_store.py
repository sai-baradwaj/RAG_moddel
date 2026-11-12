
import json
import numpy as np
import chromadb

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDING_PATH = BASE_DIR / "results" / "embeddings.npy"
METADATA_PATH = BASE_DIR / "results" / "embeddings_metadata.json"
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "hindi_rag"
MODEL_NAME = "qwen3-embedding:0.6b"


def create_vector_store():
    if not EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            "embeddings.npy not found. Run embeddings.py first."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "embeddings_metadata.json not found."
        )

    vectors = np.load(EMBEDDING_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if len(vectors) != len(metadata):
        raise ValueError(
            "Number of embeddings and metadata records do not match."
        )

    if len(vectors) == 0:
        raise ValueError("No embeddings found.")

    print("Total embeddings:", len(vectors))
    print("Total metadata:", len(metadata))
    print("Vector dimensions:", vectors.shape[1])

    client = chromadb.PersistentClient(path=str(DB_PATH))

    # Remove the old E5 collection before inserting Qwen3 vectors.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Old ChromaDB collection deleted.")
    except Exception:
        print("No previous collection found.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": MODEL_NAME
        }
    )

    collection.add(
        ids=[item["chunk_id"] for item in metadata],
        embeddings=vectors.tolist(),
        documents=[item["text"] for item in metadata],
        metadatas=[
            {"page": int(item["page"])}
            for item in metadata
        ]
    )

    print("\nVector database created successfully!")
    print("Total stored chunks:", collection.count())


if __name__ == "__main__":
    create_vector_store()
