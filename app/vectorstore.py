"""Vector store simple en Python puro.

Implementación minimalista que reemplaza ChromaDB para evitar problemas de
compilación en Windows. Usa NumPy para similitud coseno y pickle para persistir.

Para el volumen típico de un colegio (decenas a cientos de chunks), el rendimiento
es instantáneo. Si algún día necesitas escalar a miles de documentos, este módulo
se puede reemplazar por Chroma/Qdrant/FAISS sin tocar el resto del código.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Record:
    id: str
    document: str
    embedding: np.ndarray  # vector normalizado
    metadata: dict = field(default_factory=dict)


class SimpleVectorStore:
    """Almacén vectorial en memoria con persistencia en disco."""

    def __init__(self, persist_path: str | Path):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[_Record] = []
        self._load()

    # --- Persistencia -------------------------------------------------------

    def _load(self) -> None:
        if self.persist_path.exists():
            try:
                with open(self.persist_path, "rb") as f:
                    self._records = pickle.load(f)
            except Exception:
                # Si el archivo está corrupto, empezamos limpio
                self._records = []

    def _save(self) -> None:
        with open(self.persist_path, "wb") as f:
            pickle.dump(self._records, f)

    # --- API ----------------------------------------------------------------

    def count(self) -> int:
        return len(self._records)

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        if metadatas is None:
            metadatas = [{}] * len(ids)
        for i, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            vec = np.array(emb, dtype=np.float32)
            # Normalizamos para poder usar producto escalar = similitud coseno
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._records.append(_Record(id=i, document=doc, embedding=vec, metadata=meta))
        self._save()

    def query(self, query_embedding: list[float], n_results: int = 5) -> dict:
        """Devuelve los n_results más similares al embedding de consulta.

        Formato de retorno compatible con lo que usaba Chroma en el código
        original, para minimizar cambios en rag.py.
        """
        if not self._records:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "ids": [[]],
            }

        q = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # Matriz de embeddings apilados (N, dim). Similitud = producto escalar
        # (ambos vectores están ya normalizados).
        matrix = np.stack([r.embedding for r in self._records])
        similarities = matrix @ q  # shape (N,)

        # Tomamos los top-K índices (ordenados de mayor a menor similitud)
        n = min(n_results, len(self._records))
        top_idx = np.argpartition(-similarities, n - 1)[:n]
        top_idx = top_idx[np.argsort(-similarities[top_idx])]

        return {
            "documents": [[self._records[i].document for i in top_idx]],
            "metadatas": [[self._records[i].metadata for i in top_idx]],
            # "distancia" coseno = 1 - similitud (para mantener el contrato con rag.py)
            "distances": [[float(1.0 - similarities[i]) for i in top_idx]],
            "ids": [[self._records[i].id for i in top_idx]],
        }

    def reset(self) -> None:
        """Borra todos los registros y el archivo de persistencia."""
        self._records = []
        if self.persist_path.exists():
            self.persist_path.unlink()
