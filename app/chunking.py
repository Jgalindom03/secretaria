"""División del texto en fragmentos (chunks) para RAG.

Estrategia:
1. Si el documento es Markdown con encabezados (##, ###), dividimos primero
   por secciones para no mezclar temas distintos en un mismo chunk.
2. Cada sección se parte después por tokens con solapamiento si es muy larga.
3. Cada chunk lleva pegado el encabezado de su sección, para que la recuperación
   entienda a qué tema pertenece.

Esto mejora mucho la precisión con documentos bien estructurados como el del
Eurocolegio Casvi (secciones temáticas + FAQ).
"""
import re
import tiktoken
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

_encoder = tiktoken.get_encoding("cl100k_base")

# Detecta encabezados Markdown de nivel 2 y 3 (## y ###)
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Parte el texto en chunks respetando secciones Markdown cuando existen."""
    sections = _split_by_headings(text)

    chunks: list[str] = []
    for heading, body in sections:
        # Limpiamos caracteres escapados típicos de Markdown (p.ej. "1\.")
        body = body.replace("\\.", ".").strip()
        if not body:
            continue

        # Prefijo de contexto: cada chunk "recuerda" a qué sección pertenece
        prefix = f"{heading}\n\n" if heading else ""
        full = prefix + body
        tokens = len(_encoder.encode(full))

        if tokens <= chunk_size:
            chunks.append(full)
        else:
            # Sección muy larga: la partimos por párrafos manteniendo el encabezado
            prefix_tokens = len(_encoder.encode(prefix)) if prefix else 0
            sub_chunks = _chunk_long_section(body, chunk_size - prefix_tokens, overlap)
            for sc in sub_chunks:
                chunks.append(prefix + sc)

    return [c for c in chunks if c.strip()]


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Divide el texto en pares (encabezado, cuerpo).

    Si no hay encabezados, devuelve una sola sección con heading vacío.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []

    # Texto antes del primer encabezado (si hay algo relevante)
    first_start = matches[0].start()
    if first_start > 0:
        preamble = text[:first_start].strip()
        if preamble:
            sections.append(("", preamble))

    for i, m in enumerate(matches):
        heading = m.group(0).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append((heading, body))

    return sections


def _chunk_long_section(body: str, chunk_size: int, overlap: int) -> list[str]:
    """Parte una sección larga en sub-chunks por párrafos con solapamiento."""
    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(_encoder.encode(para))

        if para_tokens > chunk_size:
            if current:
                chunks.append("\n".join(current))
                current, current_tokens = [], 0
            chunks.extend(_split_by_tokens(para, chunk_size, overlap))
            continue

        if current_tokens + para_tokens <= chunk_size:
            current.append(para)
            current_tokens += para_tokens
        else:
            chunks.append("\n".join(current))
            overlap_text = _tail_overlap(current, overlap)
            current = [overlap_text, para] if overlap_text else [para]
            current_tokens = len(_encoder.encode("\n".join(current)))

    if current:
        chunks.append("\n".join(current))

    return chunks


def _split_by_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Partición a nivel de tokens para párrafos muy largos."""
    tokens = _encoder.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(_encoder.decode(tokens[start:end]))
        start = end - overlap
    return chunks


def _tail_overlap(lines: list[str], overlap_tokens: int) -> str:
    """Devuelve el final del chunk anterior para usarlo como solapamiento."""
    if not lines:
        return ""
    joined = "\n".join(lines)
    tokens = _encoder.encode(joined)
    if len(tokens) <= overlap_tokens:
        return joined
    return _encoder.decode(tokens[-overlap_tokens:])
