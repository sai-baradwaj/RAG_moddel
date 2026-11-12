
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "results" / "chunks.json"

KEYWORDS = [
    "पोखरण",
    "Pokhran",
    "परमाणु",
    "परमाणु परीक्षण",
    "आत्मकथा",
    "Wings of Fire",
    "अरुण तिवारी",
    "Arun Tiwari",
    "1999"
]

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

for chunk in chunks:
    text = chunk["text"]

    if any(keyword.lower() in text.lower() for keyword in KEYWORDS):
        print("\n" + "=" * 60)
        print("Chunk ID:", chunk["chunk_id"])
        print("Page:", chunk["page"])
        print("Text:", text)
