import urllib.request
import json
import base64
import sys

def test_ollama():
    print("Testing Ollama API...")
    url = "http://localhost:11434/api/chat"
    
    # Send a simple text message first
    payload = {
        "model": "llama3.2-vision:latest",
        "messages": [
            {"role": "user", "content": "Hello! Reply with 'OK'."}
        ],
        "stream": False
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as resp:
            print("Text Test SUCCESS:", json.loads(resp.read().decode("utf-8"))["message"]["content"])
    except Exception as e:
        print("Text Test FAILED:", e)
        if hasattr(e, 'read'):
            print("Error details:", e.read().decode("utf-8"))

    # Now test with a dummy image (1x1 red pixel JPEG)
    dummy_jpg = b'\xff\xd8\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x91\xff\xd9'
    b64_img = base64.b64encode(dummy_jpg).decode("utf-8")
    
    payload_img = {
        "model": "llama3.2-vision:latest",
        "messages": [
            {
                "role": "user", 
                "content": "Describe this image.",
                "images": [b64_img]
            }
        ],
        "stream": False
    }
    
    req_img = urllib.request.Request(
        url,
        data=json.dumps(payload_img).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    print("\nTesting with Image...")
    try:
        with urllib.request.urlopen(req_img) as resp:
            print("Image Test SUCCESS:", json.loads(resp.read().decode("utf-8"))["message"]["content"])
    except Exception as e:
        print("Image Test FAILED:", e)
        if hasattr(e, 'read'):
            print("Error details:", e.read().decode("utf-8"))

if __name__ == "__main__":
    test_ollama()
