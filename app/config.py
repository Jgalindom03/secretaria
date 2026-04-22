"""Configuración centralizada, leída desde variables de entorno."""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
SCHOOL_NAME = os.getenv("SCHOOL_NAME", "Eurocolegio Casvi")

# Parámetros de chunking
CHUNK_SIZE = 500       # tokens aprox por fragmento
CHUNK_OVERLAP = 80     # solapamiento entre fragmentos (evita cortes bruscos)

# Parámetros de recuperación
TOP_K = 5              # cuántos fragmentos recuperar por pregunta

if not OPENAI_API_KEY:
    raise RuntimeError(
        "Falta OPENAI_API_KEY. Copia .env.example a .env y añade tu clave."
    )
