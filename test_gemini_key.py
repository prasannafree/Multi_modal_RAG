import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(".env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("No GEMINI_API_KEY found in .env")
else:
    genai.configure(api_key=api_key)
    try:
        models = [m.name for m in genai.list_models()]
        print("Successfully authenticated! ALL Available models for this key:")
        for m in models:
            print(" -", m)
    except Exception as e:
        print("API Key verification failed:", e)
