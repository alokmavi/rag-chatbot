import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: No API Key found in .env")
    exit()

genai.configure(api_key=api_key)

print("--- CHECKING AVAILABLE MODELS ---")
try:
    found = False
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            print(f"AVAILABLE MODEL: {m.name}")
            found = True

    if not found:
        print("NO EMBEDDING MODELS FOUND. Check your API Key permissions.")

except Exception as e:
    print(f"ERROR CONNECTING TO GOOGLE: {e}")
