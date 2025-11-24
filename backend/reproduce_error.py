import requests
import json

def test_summarize():
    url = "http://localhost:8002/summarize"
    # The frontend likely sends the filename as the title based on previous context
    payload = {"title": "attention.pdf"}
    
    try:
        print(f"Sending request to {url} with payload: {payload}")
        response = requests.post(url, json=payload)
        
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Response JSON: {json.dumps(response.json(), indent=2)}")
        except:
            print(f"Response Text: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_summarize()
