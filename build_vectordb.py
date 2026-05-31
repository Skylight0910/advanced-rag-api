# build_vectordb.py
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PDF_PATH = "data/example.pdf"
CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def build_vectorstore():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"未找到 PDF 文件：{PDF_PATH}")

    print("📄 加载 PDF...")
    loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
    pages = loader.load()
    print(f"   共 {len(pages)} 页")

    print("✂️ 分割文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(pages)
    print(f"   得到 {len(docs)} 个文本块")

    print("🧠 向量化并存储...")
    embedding = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    # 如果目录已存在，会被覆盖
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=CHROMA_DIR
    )
    print(f"✅ 向量数据库已保存至 {CHROMA_DIR}")

if __name__ == "__main__":
    build_vectorstore()
