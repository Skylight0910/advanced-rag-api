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

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = CHUNK_SIZE
        chunk.metadata["chunk_overlap"] = CHUNK_OVERLAP

    return chunks


def main():
    print("Loading documents...")
    documents = load_text_files(DOCS_DIR)
    print(f"Loaded documents: {len(documents)}")

    if not documents:
        print("No documents found.")
        return

    print("Splitting documents...")
    chunks = split_documents(documents)
    print(f"Generated chunks: {len(chunks)}")

    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Writing chunks to Chroma...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
    )

    print("Done.")
    print(f"Chroma directory: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
