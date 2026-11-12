
from retrival import load_retrivals, retrival
from generator import generate_answer


def main():

    print("=" * 50)
    print("       HINDI DOCUMENT RAG SYSTEM")
    print("=" * 50)

    model, collection = load_retrivals()

    print("Number of stored chunks:", collection.count())
    print("System ready!")

    while True:

        question = input(
            "\nEnter your question (English/Hindi): "
        ).strip()

        if question.lower() in ["break", "exit", "quit"]:
            print("\nExiting RAG system...")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            # 1. Retrieve relevant chunks.
            retrieved_chunks = retrival(
                question,
                model,
                collection,
                top_k=10
            )

            if not retrieved_chunks:
                print("No relevant document chunks found.")
                continue

            # 2. Generate and validate evidence.
            answer = generate_answer(
                question,
                retrieved_chunks
            )

            # 3. Display grounded answer.
            print("\n" + "=" * 50)
            print("FINAL ANSWER")
            print("=" * 50)

            print(answer)

            # 4. Display retrieval diagnostics separately.
            print("\n" + "=" * 50)
            print("RETRIEVAL DIAGNOSTICS")
            print("=" * 50)

            for rank, chunk in enumerate(
                retrieved_chunks, start=1
            ):

                print(f"\nRank: {rank}")
                print("Chunk ID:", chunk["chunk_id"])
                print("Page:", chunk["metadata"]["page"])
                print("Distance:", round(chunk["distance"], 4))
                print("Text:", chunk["text"])

        except Exception as error:

            print("\nAn error occurred:")
            print(error)


if __name__ == "__main__":
    main()
