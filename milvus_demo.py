from pymilvus import MilvusClient
from langchain_huggingface import HuggingFaceEmbeddings

COLLECTION_NAME = "rag_demo_collection"
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"

docs = [
    "LangChain 适合快速构建 LLM 应用和 Agent。",
    "LangGraph 适合构建复杂、可控、有状态的多步骤 Agent 工作流。",
    "Deep Agents 提供开箱即用的 Agent 能力，包括上下文压缩、虚拟文件系统和子代理。",
    "MCP 可以将文档连接到 Claude、VSCode 等工具，用于实时文档问答。",
]

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

print("Connecting to Milvus...")
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
)

if client.has_collection(COLLECTION_NAME):
    print(f"Dropping existing collection: {COLLECTION_NAME}")
    client.drop_collection(COLLECTION_NAME)

sample_vector = embedding_model.embed_query("测试")
dim = len(sample_vector)
print("Embedding dim:", dim)

print("Creating collection...")
client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=dim,
)

print("Inserting documents...")
data = []
for i, text in enumerate(docs):
    vector = embedding_model.embed_query(text)
    data.append({
        "id": i,
        "vector": vector,
        "text": text,
    })

insert_result = client.insert(
    collection_name=COLLECTION_NAME,
    data=data,
)
print("Insert result:", insert_result)

print("Flushing collection...")
client.flush(collection_name=COLLECTION_NAME)

print("Loading collection...")
client.load_collection(collection_name=COLLECTION_NAME)

query = "如果我想要一个更可控的多步骤 Agent，应该选什么？"
query_vector = embedding_model.embed_query(query)

print("Searching...")
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    limit=3,
    output_fields=["text"],
)

print("\nQuery:", query)
print("\nRaw results:", results)
print("Result count:", len(results[0]) if results else 0)

print("\nSearch results:")
for hit in results[0]:
    print({
        "id": hit["id"],
        "distance": hit["distance"],
        "text": hit["entity"]["text"],
    })
