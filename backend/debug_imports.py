
print("Importing os...")
import os
print("Importing torch...")
import torch
print("Importing sentence_transformers...")
from sentence_transformers import SentenceTransformer
print("Importing langchain_community.embeddings...")
from langchain_community.embeddings import HuggingFaceEmbeddings
print("Importing chromadb...")
import chromadb
print("Importing langchain_community.vectorstores...")
from langchain_community.vectorstores import Chroma

print("Instantiating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Instantiating vector store with fresh dir...")
import shutil
if os.path.exists("temp_chroma"):
    shutil.rmtree("temp_chroma")
vector_store = Chroma(persist_directory="temp_chroma", embedding_function=embeddings)
# print("Skipped Chroma instantiation.")
print("Done.")
