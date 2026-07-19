from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHROMA_DB_DIR = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.3-70b-versatile"

TOP_K = 5

CHUNK_SIZE = 700

CHUNK_OVERLAP = 100

MAX_RETRIES = 2