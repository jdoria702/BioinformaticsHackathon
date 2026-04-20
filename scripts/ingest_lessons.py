import os
from app.vectorDB.ingest import extract_blocks, chunk_blocks, build_block_metadata
from app.vectorDB.retriever import ChromaRetriever

SOURCE_DIR = r"C:\Users\jdori\Desktop\bio_docs"

# Check if file is supported
def is_supported(path):
    ext = os.path.splitext(path.lower())[1]
    return ext in {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def main():
    retriever = ChromaRetriever(
        persist_dir="chroma_db",
        collection_name="bio_tutor_docs",
    )

    total_chunks = 0

    # Current directory, Subdirectories, Files
    for root, _, files in os.walk(SOURCE_DIR):
        # Access each file in the directory
        for name in files:
            path = os.path.join(root, name)
            if not is_supported(path):
                continue

            blocks, kind = extract_blocks(path)
            if not blocks:
                continue

            chunks = chunk_blocks(blocks, max_chars=1200, overlap=150)
            if not chunks:
                continue

            metadatas = build_block_metadata(
                path=path,
                kind=kind,
                block_count=len(chunks),
                ingestion_method="kb_ingest",
            )

            retriever.add_chunks(chunks, metadatas=metadatas)
            total_chunks += len(chunks)
            print(f"Ingested {name}: {len(chunks)} chunks")

    print(f"Done. Total chunks ingested: {total_chunks}")

if __name__ == "__main__":
    main()