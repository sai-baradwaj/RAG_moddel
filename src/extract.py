
import json
import re
import pymupdf
import unicodedata
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf"

OUTPUT_PATH = BASE_DIR / "results" / "extracted_pages.json"


def clean_text(text):

    # Normalize Unicode characters
    text = unicodedata.normalize("NFC", text)

    # Normalize Windows and other newline characters
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def extract_pdf():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Place the dataset in the data folder."
        )

    document = pymupdf.open(PDF_PATH)

    pages = []

    print(f"Total PDF pages: {len(document)}")

    for page_index, page in enumerate(document):

        # Extract text without TEXT_INHIBIT_SPACES
        raw_text = page.get_text(
            "text",
            sort=True
        )

        cleaned_text = clean_text(raw_text)

        page_data = {
            "page": page_index + 1,
            "text": cleaned_text
        }

        pages.append(page_data)

        print(
            f"Extracted page {page_index + 1} "
            f"| Characters: {len(cleaned_text)}"
        )

    document.close()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

        json.dump(
            pages,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nPDF extraction completed!")
    print(f"Output saved to: {OUTPUT_PATH}")

    if pages:
        print("\nFirst page preview:")
        print(pages[0]["text"][:1000])

    return pages


if __name__ == "__main__":
    extract_pdf()
