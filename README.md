# RAG over a Hindi Document

**Assignment:** Internship Screening Assignment – Level 1  
**Organization:** MegaMindz AI Solutions Pvt Ltd  
**Document:** `Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf`

## 1. Project overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline over a Hindi-language PDF about Dr. A. P. J. Abdul Kalam. It extracts and normalizes document text, divides it into chunks, creates multilingual embeddings, stores the vectors in ChromaDB, retrieves relevant chunks for Hindi or English questions, and uses a locally hosted Ollama model to generate answers.

## 2. Architecture

```text
Hindi PDF
   |
   v
PyMuPDF text extraction + Unicode normalization
   |
   v
Paragraph/sentence-aware chunking
   |
   v
Qwen3-Embedding-0.6B embeddings
   |
   v
ChromaDB (persistent collection: hindi_rag)
   |
   v
Hindi/English query embedding
   |
   v
Top-k semantic retrieval
   |
   v
Ollama (Qwen3:8B) answer generation
   |
   v
Answer + retrieved source metadata
```

## 3. Technology stack

- **PDF extraction:** PyMuPDF (`pymupdf`)
- **Text normalization:** Python `unicodedata` (NFC) and regular expressions
- **Chunking:** paragraph/sentence-aware chunks (configured in `src/chunking.py`)
- **Embedding model:** `qwen3-embedding:0.6b` (served locally through Ollama)
- **Vector database:** ChromaDB, persistent local storage
- **LLM runtime:** Ollama
- **Generation model:** `qwen3:8b`
- **Application:** Python command-line interface

## 4. Project structure

```text
Hindi-RAG-Project/
├── data/
│   └── Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf
├── src/
│   ├── extract.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrival.py
│   ├── generator.py
│   ├── main.py
│   └── search_chunks.py       # optional debugging utility
├── results/
│   ├── extracted_pages.json
│   ├── chunks.json
│   ├── embeddings.npy
│   └── embeddings_metadata.json
├── chroma_db/                 # generated persistent vector database
├── requirements.txt
├── evaluation_results.json
└── README.md
```

The exact metadata JSON filename may differ according to the filename configured in `src/embeddings.py`. Keep the configured filename consistent with `src/vector_store.py`.

## 5. Setup

### Prerequisites

1. Python 3.10 or a compatible Python version.
2. Ollama installed and running.
3. The Qwen3 8B model downloaded in Ollama.
4. The supplied PDF placed in the `data/` directory.

Download the model once:

```powershell
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

Start Ollama if it is not already running:

```powershell
ollama serve
```

### Install Python dependencies

From the project root, activate your virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The Python packages used by the current implementation are PyMuPDF, ChromaDB, NumPy and Requests. Embeddings are generated through the local Ollama API; `sentence-transformers` is not required. Ensure `requirements.txt` contains the packages actually imported by your source files.

## 6. Build the RAG index

Run these commands from the project root, in order:

```powershell
python .\src\extract.py
python .\src\chunking.py
python .\src\embeddings.py
python .\src\vector_store.py
```

These steps extract PDF text, create chunks, generate Qwen3 embeddings through Ollama, and persist the chunks in ChromaDB. Before running `embeddings.py`, ensure Ollama is running and `qwen3-embedding:0.6b` is pulled.

If you change chunk content or metadata, rebuild the embeddings and vector database. Do not reuse an old ChromaDB index with newly generated chunks. If the collection already exists and contains stale records, delete/recreate the `hindi_rag` collection or clear the local `chroma_db` directory before rebuilding.

## 7. Run the application

Keep Ollama running, open another PowerShell terminal at the project root, activate the virtual environment, and run:

```powershell
python .\src\main.py
```

Enter a question in Hindi or English. Enter `break` to exit the CLI.

## 8. Retrieval and multilingual handling

The application uses `qwen3-embedding:0.6b` through Ollama's `/api/embed` endpoint. Document chunks are embedded as text, and queries use an instruction prefix requesting relevant passages from a Hindi document. Both document and query vectors are L2-normalized before cosine-similarity search. The same embedding model must be used for indexing and querying.

The query is embedded and submitted to ChromaDB, which returns the top-k chunks with their text, metadata and distance values.

## 9. Chunking strategy

The implementation uses paragraph/sentence-aware chunking with a configured maximum chunk length and overlap. This keeps related sentences together while limiting the amount of context sent to the language model. The exact chunk size and overlap should be reported from the values configured in `src/chunking.py`.

## 10. Citations

Retrieved chunk metadata includes the source PDF page and chunk ID. The CLI displays the retrieved page, chunk ID and vector distance. These identify the retrieved evidence supplied to the generator.

Important limitation: displaying all retrieved chunks is not the same as proving that each chunk supports every generated claim. The current prompt-based generator can still make unsupported statements or produce an incorrect citation in its answer text. Review the evaluation results and do not treat an LLM-written citation as validated unless it matches a retrieved chunk and its text supports the claim.

## 11. Evaluation

The required six assignment questions were run through the CLI. The observed outcomes are recorded in `evaluation_results.json`. The current test run demonstrates that the pipeline executes and answers some questions, but answer grounding and citation reliability still need improvement. In particular, the aeronautical-engineering response confused MIT with IIT, the Pokhran-II answer was vague, and the death question was not answered from the retrieved context.

## 12. Current limitations and future improvements

- Improve Devanagari PDF extraction where the extracted text contains spacing or character artifacts.
- Improve general-purpose retrieval for facts that are not ranked in the top-k results.
- Add a general-purpose reranker or hybrid lexical/vector retrieval if evaluation shows it is needed.
- Validate answer citations programmatically against retrieved chunk IDs and supporting text.
- Extract section headings if section-level citations are desired.
- Evaluate retrieval with Recall@k / MRR and answer quality with a manually checked question set.

## 13. Notes

- The current implementation uses PyMuPDF text extraction. OCR is not part of the active pipeline.
- The project is a command-line RAG application; a web frontend is not required by the assignment.
- Do not commit virtual environments, model cache directories, or unnecessary temporary files to the submission ZIP.


## 14. Reproducibility note for submission

The ChromaDB directory is intentionally not required in the submission archive. Rebuild it locally using the commands in Section 6. The embedding model must be `qwen3-embedding:0.6b` for both indexing and querying; do not mix vectors from a different embedding model.

The `evaluation_results.json` file records earlier baseline observations and should not be interpreted as a completed evaluation of the current Qwen3 embedding configuration. The current implementation still requires a fresh six-question evaluation after rebuilding the index.
