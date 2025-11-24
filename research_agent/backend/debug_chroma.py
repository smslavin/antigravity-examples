from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def debug_chroma():
    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="../chroma_db", embedding_function=embeddings)
    
    print(f"Collection count: {vector_store._collection.count()}")
    
    print("Peeking at first 3 items:")
    results = vector_store._collection.get(limit=3, include=["metadatas", "documents"])
    for i, meta in enumerate(results["metadatas"]):
        print(f"Item {i} metadata: {meta}")
        # print(f"Item {i} content: {results['documents'][i][:100]}...")

    print("\nTesting retrieval for 'attention.pdf'...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke("attention.pdf")
    print(f"Retrieved {len(docs)} docs for query 'attention.pdf'")
    for d in docs:
        print(f" - Source: {d.metadata.get('source')}")

if __name__ == "__main__":
    debug_chroma()
