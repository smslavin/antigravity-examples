from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

def inspect():
    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="../chroma_db_new", embedding_function=embeddings)
    
    print("Getting all documents...")
    # Get all ids to retrieve all metadata
    result = vector_store.get()
    metadatas = result['metadatas']
    
    unique_sources = set()
    for m in metadatas:
        if 'source' in m:
            unique_sources.add(m['source'])
            
    print("\n--- Unique Sources in Vector Store ---")
    for s in unique_sources:
        print(f"'{s}'")

if __name__ == "__main__":
    inspect()
