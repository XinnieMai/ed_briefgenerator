"""
Configuration file for AURA-ED project. 
This files contains all the necessary configurations for the projects.

Includes
- LLM configuration
- Requirments.txt 
"""
import os

from dotenv import load_dotenv
import ollama
from google import genai

load_dotenv()
print("Loading configuration...")

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
AUTO_PULL = os.getenv("AUTO_PULL", "true").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")

ollama_client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)


# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3-flash-preview"

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# Auto download model
def _model_available():
    try:
        ollama_client.show(OLLAMA_MODEL)
        return True
    except ollama.ResponseError:
        return False
    
if AUTO_PULL and not _model_available():
    print(f"[AURA-ED] Pulling Ollama model '{OLLAMA_MODEL}'...")
    ollama_client.pull(OLLAMA_MODEL)
    print(f"[AURA-ED] Ollama model '{OLLAMA_MODEL}' ready.")
else:
    print(f"[AURA-ED] Ollama default model '{OLLAMA_MODEL}' found locally.")
 
 
# Load requirements   
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()    