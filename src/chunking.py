
import json
import re
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_PATH / "results" / "extracted_pages.json"
OUTPUT_PATH = BASE_PATH / "results" / "chunks.json"

MAX_CHUNK = 500


def split_sentences(paragraph):

    # Normalize whitespace inside the paragraph
    paragraph = re.sub(r"\s+", " ", paragraph).strip()

    # Split after Hindi danda or English sentence punctuation
    sentences = re.split(r"(?<=[।.!?])\s+", paragraph)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def create_chunks(page_data):

    text = page_data["text"]
    page_number = page_data["page"]

    # Split text into paragraphs
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current_sentences = []

    for paragraph in paragraphs:

        if not paragraph.strip():
            continue

        sentences = split_sentences(paragraph)

        for sentence in sentences:

            candidate = " ".join(current_sentences + [sentence])

            # Check whether adding this sentence exceeds the limit
            if current_sentences and len(candidate) > MAX_CHUNK:

                # Save the current chunk
                chunks.append({
                    "text": " ".join(current_sentences)
                })

                # Keep the last sentence as overlap when possible
                previous_sentence = current_sentences[-1]

                overlap_candidate = previous_sentence + " " + sentence

                if len(overlap_candidate) <= MAX_CHUNK:
                    current_sentences = [
                        previous_sentence,
                        sentence
                    ]
                else:
                    current_sentences = [sentence]

            else:
                current_sentences.append(sentence)

    # Save the remaining sentences
    if current_sentences:
        chunks.append({
            "text": " ".join(current_sentences)
        })

    # Attach page number and chunk ID placeholder
    final_chunks = []

    for chunk in chunks:

        final_chunks.append({
            "chunk_id": None,
            "page": page_number,
            "text": chunk["text"]
        })

    return final_chunks


def process_document():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Extracted pages JSON not found. Run extract.py first."
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        pages = json.load(file)

    all_chunks = []

    for page_data in pages:

        page_chunks = create_chunks(page_data)

        all_chunks.extend(page_chunks)

    # Assign unique chunk IDs
    for index, chunk in enumerate(all_chunks):

        chunk["chunk_id"] = f"chunk_{index + 1:04d}"

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

        json.dump(
            all_chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nChunking completed!")
    print(f"Total pages processed: {len(pages)}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Maximum chunk size: {MAX_CHUNK} characters")
    print("Overlap: One sentence, when it fits")
    print(f"Output saved to: {OUTPUT_PATH}")

    if all_chunks:

        print("\nFirst chunk:\n")

        print(
            json.dumps(
                all_chunks[0],
                ensure_ascii=False,
                indent=4
            )
        )

    return all_chunks


if __name__ == "__main__":
    process_document()
