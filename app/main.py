"""API FastAPI para la interfaz de preguntas frecuentes del colegio."""
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.rag import ingest_file, answer_question, answer_question_stream, clear_collection, get_store
from app.config import SCHOOL_NAME

app = FastAPI(title=f"FAQ {SCHOOL_NAME}", version="1.0.0")

# Servimos el frontend estático
STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- Schemas -----------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    source: str
    excerpt: str
    similarity: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


# --- Rutas -------------------------------------------------------------------

@app.get("/")
def root():
    """Sirve el frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
def admin():
    """Sirve el panel de administración para subir documentos."""
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/api/health")
def health():
    store = get_store()
    return {
        "status": "ok",
        "school": SCHOOL_NAME,
        "chunks_indexed": store.count(),
    }


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    """Sube un documento (PDF/DOCX/TXT) y lo añade a la base de conocimiento."""
    allowed = {".pdf", ".docx", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado. Usa uno de: {', '.join(allowed)}",
        )

    # Guardamos el upload en un temporal y lo procesamos
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        stats = ingest_file(tmp_path, source_name=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return stats


@app.post("/api/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    """Responde a una pregunta del usuario usando RAG."""
    try:
        result = answer_question(req.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar respuesta: {e}")


@app.post("/api/ask/stream")
def ask_stream(req: QuestionRequest):
    """Streaming version of /api/ask using Server-Sent Events."""
    return StreamingResponse(
        answer_question_stream(req.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/reset")
def reset():
    """Borra toda la base de conocimiento. Pensado para el admin."""
    clear_collection()
    return {"status": "ok", "message": "Base de conocimiento borrada."}
