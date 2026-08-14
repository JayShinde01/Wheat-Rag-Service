import os

CHROMA_URL = os.getenv(
    "CHROMA_URL",
    "https://wheat-cromadb.onrender.com"
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "wheat_knowledge"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

PORT = int(os.getenv("PORT", "5001"))