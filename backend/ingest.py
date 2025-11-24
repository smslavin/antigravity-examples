import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Global components to avoid reloading
embeddings = None
vector_store = None

def get_vector_store():
    global embeddings, vector_store
    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory="../chroma_db_new", embedding_function=embeddings)
    return vector_store

def ingest_pdf(file_path):
    print(f"Ingesting {file_path}...")
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Add source metadata if missing (PyPDFLoader usually adds it)
        for chunk in chunks:
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = file_path
                
        vs = get_vector_store()
        vs.add_documents(chunks)
        print(f"Successfully ingested {len(chunks)} chunks from {file_path}")
        return True
    except Exception as e:
        print(f"Error ingesting {file_path}: {e}")
        return False
