from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import threading
import json
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from custom_llm import CustomGeminiLLM
from watcher import start_watcher

load_dotenv()

app = FastAPI(title="Research Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
papers_dir = "../papers"
abs_papers_dir = os.path.abspath(papers_dir)
print(f"Mounting static files from: {abs_papers_dir}")
if os.path.exists(abs_papers_dir):
    print(f"Files in papers dir: {os.listdir(abs_papers_dir)}")
else:
    print("Papers dir does not exist!")

if not os.path.exists(papers_dir):
    os.makedirs(papers_dir)
app.mount("/pdfs", StaticFiles(directory=papers_dir), name="pdfs")

# Initialize components
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="../chroma_db_new", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = CustomGeminiLLM(model_name="gemini-flash-latest", google_api_key=GOOGLE_API_KEY)
search_tool = DuckDuckGoSearchRun()

# Start watcher in background thread
def run_watcher():
    papers_dir = "../papers"
    if not os.path.exists(papers_dir):
        os.makedirs(papers_dir)
    start_watcher(papers_dir)

watcher_thread = threading.Thread(target=run_watcher, daemon=True)
watcher_thread.start()

class SummarizeRequest(BaseModel):
    title: str

class SearchRequest(BaseModel):
    query: str

# Load metadata
METADATA_FILE = "metadata.json"
metadata_lock = threading.Lock()

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_metadata(data):
    with metadata_lock:
        with open(METADATA_FILE, "w") as f:
            json.dump(data, f, indent=2)

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    try:
        # Check cache first
        metadata = load_metadata()
        if request.title in metadata:
            print(f"Returning cached summary for {request.title}")
            return {"summary": metadata[request.title]["summary"]}

        # Use metadata filtering to find the specific paper
        # The source metadata usually contains the full path, e.g., "../papers/attention.pdf"
        # We'll try to match the filename
        
        # First, try to find documents with matching source in metadata
        # Since we don't know the exact path stored, we might need to do a search or just rely on the retriever with filter
        
        # Construct a filter for the source
        # Note: Chroma filter syntax might vary, but usually it's where={"source": "path"}
        # But we only have the filename from the frontend.
        # Let's try to construct the relative path used in ingestion
        # IMPORTANT: Handle both forward and backslashes due to Windows/Linux differences in ingestion
        # The vector store might have mixed paths like "../papers\file.pdf"
        
        possible_paths = [
            os.path.normpath(os.path.join("../papers", request.title)), # Windows: ..\papers\file.pdf
            f"../papers/{request.title}",                               # POSIX: ../papers/file.pdf
            f"../papers\\{request.title}"                               # Mixed: ../papers\file.pdf
        ]
        
        docs_content = []
        for path in possible_paths:
            print(f"Attempting to retrieve with source: {path}")
            results = vector_store.get(where={"source": path})
            if results['documents']:
                docs_content = results['documents']
                print(f"Found match with path: {path}")
                break
        
        if not docs_content:
            print(f"Failed to find document with source matching {request.title}. Tried: {possible_paths}")
            return {"summary": "Error: Could not find document content in library. Please ensure the file is ingested."}
        
        # Limit context to avoid token limits (e.g. first 30k chars)
        context = "\n\n".join(docs_content)[:30000]
        
        template = """You are a research assistant.
        First, identify the real title of the paper from the provided context.
        Then, summarize the paper.
        
        Context:
        {context}
        
        Format your response exactly as follows:
        **Title: [Real Title]**
        
        [Summary]"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        
        summary = chain.invoke({"context": context})
        
        # Save to cache
        metadata = load_metadata() # Reload in case it changed
        metadata[request.title] = {
            "summary": summary,
            "timestamp": time.time()
        }
        save_metadata(metadata)
        
        return {"summary": summary}
    except Exception as e:
        print(f"Error in summarize: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(request: SearchRequest):
    try:
        results = search_tool.invoke(request.query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers")
async def list_papers():
    # List files in papers directory
    papers_dir = "../papers"
    if not os.path.exists(papers_dir):
        return {"papers": []}
    files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
    return {"papers": files}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
