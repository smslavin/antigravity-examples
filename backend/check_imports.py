try:
    import watchdog
    print("watchdog imported")
    import langchain
    print("langchain imported")
    import chromadb
    print("chromadb imported")
    import fastapi
    print("fastapi imported")
    import sentence_transformers
    print(f"sentence_transformers imported from {sentence_transformers.__file__}")
    import duckduckgo_search
    print("duckduckgo_search imported")
    
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("HuggingFaceEmbeddings imported")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("HuggingFaceEmbeddings instantiated")
    
    print("All imports successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
