from langchain_community.document_loaders import PyPDFLoader
import os

def test_extraction():
    file_path = "../papers/attention.pdf"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    if not docs:
        print("No documents loaded.")
        return

    first_page = docs[0]
    print("--- Metadata ---")
    print(first_page.metadata)
    
    print("\n--- First Page Content (First 500 chars) ---")
    print(first_page.page_content[:500])
    
    # Try to extract title using LLM if metadata is insufficient
    # We can simulate this by printing what we would send to the LLM
    print("\n--- LLM Prompt for Title Extraction ---")
    print(f"Extract the paper title from the following text:\n{first_page.page_content[:1000]}")

if __name__ == "__main__":
    test_extraction()
