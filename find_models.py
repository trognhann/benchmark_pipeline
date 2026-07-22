import urllib.request
import json
import sys

def search_hf(query):
    url = f"https://huggingface.co/api/models?search={query}&limit=5"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            print(f"Search results for '{query}':")
            for model in data:
                print(f" - {model['id']}")
            print("")
    except Exception as e:
        print(f"Failed to search for '{query}': {e}")

if __name__ == "__main__":
    search_hf("CartoonGAN ONNX")
    search_hf("CartoonGAN")
    search_hf("yolov8n-face")
