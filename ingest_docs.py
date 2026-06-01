import hashlib
from collections import defaultdict
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path("docs")
CHROMA_DIR = "./chroma_db"

EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_text_files(docs_dir: Path) -> list[Document]:
    documents = []

    for path in docs_dir.rglob("*"):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "file_type": path.suffix.lower(),
                },
            )
        )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    chunk_counts = defaultdict(int)

    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk.metadata["chunk_id"] = chunk_counts[source]
        chunk.metadata["chunk_size"] = CHUNK_SIZE
        chunk.metadata["chunk_overlap"] = CHUNK_OVERLAP
        chunk_counts[source] += 1

    return chunks


def build_chunk_id(chunk: Document) -> str:
    source = chunk.metadata["source"]
    content = chunk.page_content
    value = f"{source}\n{content}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def delete_stale_chunks(
    vectordb: Chroma,
    managed_sources: set[str],
    current_ids: set[str],
) -> int:
    existing = vectordb.get(include=["metadatas"])
    stale_ids = []

    for record_id, metadata in zip(
        existing.get("ids", []),
        existing.get("metadatas", []),
    ):
        metadata = metadata or {}
        if (
            metadata.get("source") in managed_sources
            and record_id not in current_ids
        ):
            stale_ids.append(record_id)

    if stale_ids:
        vectordb.delete(ids=stale_ids)

    return len(stale_ids)


def main():
    print("Loading documents...")
    documents = load_text_files(DOCS_DIR)
    print(f"Loaded documents: {len(documents)}")

    if not documents:
        print("No documents found.")
        return

    print("Splitting documents...")
    chunks = split_documents(documents)
    chunk_ids = [build_chunk_id(chunk) for chunk in chunks]
    print(f"Generated chunks: {len(chunks)}")

    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
    )

    managed_sources = {
        document.metadata["source"]
        for document in documents
    }

    deleted_count = delete_stale_chunks(
        vectordb,
        managed_sources,
        set(chunk_ids),
    )
    print(f"Deleted stale chunks: {deleted_count}")

    print("Writing chunks to Chroma...")
    vectordb.add_documents(
        documents=chunks,
        ids=chunk_ids,
    )

    print("Done.")
    print(f"Upserted chunks: {len(chunks)}")
    print(f"Chroma directory: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
