"""Motor RAG: indexa documentos y responde preguntas.

Flujo:
1. Al ingerir: extraer texto -> dividir en chunks -> embeddings -> guardar en el vector store
2. Al preguntar: embedding de la pregunta -> buscar top-K chunks -> LLM responde

Vector store: usamos un SimpleVectorStore en Python puro para evitar problemas
de compilación en Windows. Para el volumen típico de un colegio va sobrado.

Nota sobre proveedores: usamos OpenAI. Si prefieres Mistral, solo tienes
que cambiar el cliente y los nombres de modelo. Mistral ofrece:
  - embeddings:  mistral-embed
  - chat:        mistral-small-latest (o mistral-large-latest)
La API es muy similar (from mistralai import Mistral).
"""
from __future__ import annotations
import uuid
from pathlib import Path

from openai import OpenAI

from app.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    CHROMA_PATH,
    TOP_K,
    SCHOOL_NAME,
)
from app.loaders import extract_text
from app.chunking import chunk_text
from app.vectorstore import SimpleVectorStore


# --- Clientes singleton ------------------------------------------------------

_openai_client: OpenAI | None = None
_vector_store: SimpleVectorStore | None = None


def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def get_store() -> SimpleVectorStore:
    """Devuelve el vector store, creándolo/cargándolo si hace falta."""
    global _vector_store
    if _vector_store is None:
        persist_file = Path(CHROMA_PATH) / "vectorstore.pkl"
        _vector_store = SimpleVectorStore(persist_file)
    return _vector_store


# --- Embeddings --------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de textos en una sola llamada."""
    if not texts:
        return []
    response = get_openai().embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


# --- Ingesta -----------------------------------------------------------------

def ingest_file(file_path: str, source_name: str | None = None) -> dict:
    """Indexa un archivo completo en la base de datos vectorial.

    Args:
        file_path: ruta al archivo (PDF, DOCX, TXT, MD).
        source_name: nombre legible para mostrar al citar (por defecto, el nombre del archivo).

    Returns:
        dict con estadísticas de la ingesta.
    """
    source = source_name or Path(file_path).name

    text = extract_text(file_path)
    if not text.strip():
        raise ValueError("El documento no contiene texto extraíble.")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No se pudieron generar fragmentos del documento.")

    embeddings = embed_texts(chunks)

    store = get_store()
    ids = [f"{source}-{uuid.uuid4().hex[:8]}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

    store.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {
        "source": source,
        "chunks_added": len(chunks),
        "total_in_collection": store.count(),
    }


def clear_collection() -> None:
    """Borra toda la base de conocimiento. Útil para reindexar desde cero."""
    global _vector_store
    store = get_store()
    store.reset()
    _vector_store = None  # forzamos recarga la próxima vez


# --- Consulta (RAG) ----------------------------------------------------------

# --- System prompt anterior --------------------------------------------------
# SYSTEM_PROMPT = """Eres el asistente virtual de {school}. Tu trabajo es responder a preguntas frecuentes de familias, alumnos y personal a partir EXCLUSIVAMENTE del contexto que se te proporciona, extraído de la documentación oficial del colegio.
#
# Reglas:
# - Si la respuesta NO está claramente en el contexto, di honestamente que no dispones de esa información y sugiere contactar con secretaría del colegio. No inventes datos.
# - Responde en español, con un tono cercano, claro y profesional.
# - Sé conciso: ve al grano, usa listas si ayudan a la claridad.
# - No inventes horarios, precios, nombres de profesores ni fechas.
# - Si el usuario pregunta algo ajeno al colegio, redirígelo amablemente."""
#
# --- System prompt activo ----------------------------------------------------

SYSTEM_PROMPT = """Eres la secretaría virtual del Eurocolegio Casvi (Castillo de Villaviciosa, Madrid), un centro educativo concertado de alto nivel con programa del Bachillerato Internacional.

Tu función es atender las consultas de familias, alumnos y personal a partir EXCLUSIVAMENTE del contexto extraído de la documentación oficial del colegio.

Normas de comportamiento:
- Responde siempre usando la información del contexto proporcionado. No inventes datos, fechas, precios ni nombres.
- Si la respuesta no está en el contexto, indícalo con claridad y sugiere al usuario que contacte directamente con secretaría en info@casviboadilla.es o en el teléfono (+34) 91 632 96 53.
- Tono: amable, profesional e institucional. Nunca informal ni robótico.
- Formato: usa Markdown para estructurar la respuesta cuando sea útil (listas, negritas, tablas). Evita párrafos innecesariamente largos.
- Idioma: responde en el mismo idioma en que se formula la pregunta.
- No hagas suposiciones sobre información no presente en el contexto."""


def answer_question_stream(question: str, top_k: int = TOP_K):
    """Streaming RAG response as SSE-formatted events.

    Yields:
        data: {"type": "token",   "content": "..."}
        data: {"type": "sources", "sources": [...]}
        data: {"type": "done"}
        data: {"type": "error",   "content": "..."}  (on failure)
    """
    import json

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    if not question.strip():
        yield sse({"type": "token", "content": "Por favor, formule su consulta."})
        yield sse({"type": "done"})
        return

    store = get_store()
    if store.count() == 0:
        yield sse({"type": "token", "content": (
            "Aún no se ha cargado documentación del colegio en el sistema. "
            "Por favor, contacte con el equipo de administración."
        )})
        yield sse({"type": "done"})
        return

    try:
        query_embedding = embed_texts([question])[0]

        results = store.query(
            query_embedding=query_embedding,
            n_results=min(top_k, store.count()),
        )

        retrieved_docs: list[str] = results["documents"][0]
        retrieved_meta: list[dict] = results["metadatas"][0]
        retrieved_dist: list[float] = results["distances"][0]

        context_blocks = []
        for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_meta), start=1):
            context_blocks.append(
                f"[Fragmento {i} — fuente: {meta.get('source', 'documento')}]\n{doc}"
            )
        context = "\n\n---\n\n".join(context_blocks)

        user_message = (
            f"CONTEXTO DE LA DOCUMENTACIÓN DEL COLEGIO:\n\n{context}\n\n"
            f"PREGUNTA DEL USUARIO: {question}\n\n"
            f"Responde usando solo la información del contexto anterior."
        )

        stream = get_openai().chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield sse({"type": "token", "content": delta.content})

        sources = []
        for doc, meta, dist in zip(retrieved_docs, retrieved_meta, retrieved_dist):
            sources.append({
                "source": meta.get("source", "documento"),
                "excerpt": doc[:300] + ("..." if len(doc) > 300 else ""),
                "similarity": round(1 - dist, 3),
            })

        yield sse({"type": "sources", "sources": sources})
        yield sse({"type": "done"})

    except Exception as e:
        yield sse({"type": "error", "content": (
            "Ha ocurrido un error al procesar su consulta. "
            "Por favor, inténtelo de nuevo o contacte con secretaría."
        )})
        yield sse({"type": "done"})


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    """Responde a una pregunta usando RAG.

    Returns:
        dict con 'answer' (texto) y 'sources' (lista de fragmentos usados).
    """
    if not question.strip():
        return {"answer": "Por favor, escribe una pregunta.", "sources": []}

    store = get_store()
    if store.count() == 0:
        return {
            "answer": (
                "Aún no se ha cargado ningún documento con información del colegio. "
                "Sube primero el documento desde el panel de administración."
            ),
            "sources": [],
        }

    # 1. Embedding de la pregunta
    query_embedding = embed_texts([question])[0]

    # 2. Recuperación
    results = store.query(
        query_embedding=query_embedding,
        n_results=min(top_k, store.count()),
    )

    retrieved_docs: list[str] = results["documents"][0]
    retrieved_meta: list[dict] = results["metadatas"][0]
    retrieved_dist: list[float] = results["distances"][0]

    # 3. Construcción del contexto con marcadores para que el modelo pueda referenciarlos
    context_blocks = []
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_meta), start=1):
        context_blocks.append(f"[Fragmento {i} — fuente: {meta.get('source', 'documento')}]\n{doc}")
    context = "\n\n---\n\n".join(context_blocks)

    user_message = (
        f"CONTEXTO DE LA DOCUMENTACIÓN DEL COLEGIO:\n\n{context}\n\n"
        f"PREGUNTA DEL USUARIO: {question}\n\n"
        f"Responde usando solo la información del contexto anterior."
    )

    # 4. Llamada al LLM
    chat_response = get_openai().chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,   # baja para respuestas factuales
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(school=SCHOOL_NAME)},
            {"role": "user", "content": user_message},
        ],
    )

    answer = chat_response.choices[0].message.content

    # 5. Preparamos las fuentes para enseñarlas en el frontend
    sources = []
    for doc, meta, dist in zip(retrieved_docs, retrieved_meta, retrieved_dist):
        sources.append({
            "source": meta.get("source", "documento"),
            "excerpt": doc[:300] + ("..." if len(doc) > 300 else ""),
            "similarity": round(1 - dist, 3),  # distancia coseno -> similitud
        })

    return {"answer": answer, "sources": sources}
