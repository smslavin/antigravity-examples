import requests
import json
import time
import os

def verify_persistence():
    url = "http://localhost:8002/summarize"
    payload = {"title": "attention.pdf"}
    metadata_file = "metadata.json"
    
    # Clear metadata for fresh test
    if os.path.exists(metadata_file):
        with open(metadata_file, "w") as f:
            json.dump({}, f)
            
    print("--- Request 1 (Generating) ---")
    start = time.time()
    response1 = requests.post(url, json=payload)
    duration1 = time.time() - start
    print(f"Status: {response1.status_code}, Time: {duration1:.2f}s")
    print(f"Summary Start: {response1.json()['summary'][:50]}...")
    
    # Check if metadata file is updated
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            data = json.load(f)
            if "attention.pdf" in data:
                print("\n[PASS] Metadata file updated.")
            else:
                print("\n[FAIL] Metadata file NOT updated.")
    else:
        print("\n[FAIL] Metadata file not found.")

    print("\n--- Request 2 (Cached) ---")
    start = time.time()
    response2 = requests.post(url, json=payload)
    duration2 = time.time() - start
    print(f"Status: {response2.status_code}, Time: {duration2:.2f}s")
    
    if duration2 < 2.0:
        print("\n[PASS] Response time indicates cache hit.")
    else:
        print("\n[WARN] Response time slow, might not be cached.")

    if response1.json()['summary'] == response2.json()['summary']:
        print("[PASS] Summaries match.")
    else:
        print("[FAIL] Summaries do not match.")

if __name__ == "__main__":
    verify_persistence()
