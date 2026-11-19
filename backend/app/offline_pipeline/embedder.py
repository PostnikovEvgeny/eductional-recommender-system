import gc
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.core.config import ROOT_DIR, settings

logger = logging.getLogger(__name__)


class FridaEmbedder:
    """
    Оптимизированный сервис управления локальной моделью FRIDA.
    Обеспечивает быстрый батч-инференс и L2-нормализацию всех векторов.
    """

    def __init__(
        self,
        model_name: str = settings.model_name,
        local_dir: str | Path = settings.local_model_path,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.local_path = (
            Path(local_dir)
            if Path(local_dir).is_absolute()
            else (ROOT_DIR / local_dir).resolve()
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: SentenceTransformer | None = None

    def _is_model_downloaded(self) -> bool:
        """Проверяет, сохранены ли веса и конфигурация модели на локальном диске."""
        config_file = self.local_path / "config.json"
        return self.local_path.exists() and config_file.is_file()

    def load_model(self) -> SentenceTransformer:
        """Загружает модель с диска и настраивает оптимальное число потоков CPU."""
        if self._model is not None:
            return self._model

        self.local_path.mkdir(parents=True, exist_ok=True)

        if self.device == "cpu":
            cpu_count = os.cpu_count() or 4
            torch.set_num_threads(max(4, cpu_count))

        if self._is_model_downloaded():
            logger.info(
                "Загрузка локальной модели FRIDA с диска: %s (device=%s)",
                self.local_path,
                self.device,
            )
            model = SentenceTransformer(str(self.local_path), device=self.device)
        else:
            logger.info(
                "Скачивание '%s' из Hugging Face в %s...",
                self.model_name,
                self.local_path,
            )
            model = SentenceTransformer(self.model_name, device=self.device)
            model.save(str(self.local_path))

        model.eval()
        self._model = model
        gc.collect()
        return self._model

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        prefix: str = "search_document: ",
        batch_size: int = 64,
    ) -> np.ndarray:
        """Батч-кодирование списка текстов с отключенным графом градиентов."""
        if not texts:
            return np.empty((0, settings.embedding_dimension), dtype=np.float32)

        model = self.load_model()
        prefixed_texts = [f"{prefix}{t}" if prefix else t for t in texts]

        with torch.inference_mode():
            embeddings = model.encode(
                prefixed_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_documents(
        self, documents: Sequence[str], batch_size: int = 64
    ) -> np.ndarray:
        """Кодирует тексты документов (префикс 'search_document: ')."""
        return self.encode_texts(
            documents, prefix="search_document: ", batch_size=batch_size
        )

    def encode_queries(
        self, queries: Sequence[str], batch_size: int = 64
    ) -> np.ndarray:
        """Кодирует названия интересов / доменов (префикс 'search_query: ')."""
        return self.encode_texts(
            queries, prefix="search_query: ", batch_size=batch_size
        )

    def split_into_chunks(
        self,
        document: str,
        *,
        chunk_size: int = 2000,
        overlap: int = 200,
        max_chunks: int = 2,
    ) -> list[str]:
        """Разбивает длинный текст на не более чем max_chunks смысловых фрагментов."""
        if not document:
            return []
        if len(document) <= chunk_size:
            return [document]

        chunks: list[str] = []
        step = max(1, chunk_size - overlap)
        for start in range(0, len(document), step):
            chunk = document[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if len(chunks) >= max_chunks:
                break
        return chunks

    def encode_long_document(
        self,
        document: str,
        *,
        prefix: str = "search_document: ",
        chunk_size: int = 2000,
        overlap: int = 200,
        max_chunks: int = 2,
    ) -> np.ndarray:
        """Кодирует длинный текст аннотации по перекрывающимся чанкам."""
        chunks = self.split_into_chunks(
            document, chunk_size=chunk_size, overlap=overlap, max_chunks=max_chunks
        )
        if not chunks:
            return np.zeros(settings.embedding_dimension, dtype=np.float32)

        chunk_vecs = self.encode_texts(chunks, prefix=prefix)
        mean_vec = np.mean(chunk_vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm

        return np.asarray(mean_vec, dtype=np.float32)

    def calculate_direction_centroid(
        self, discipline_embeddings: np.ndarray
    ) -> np.ndarray:
        """Вычисляет интегральный вектор направления подготовки V_D (формула 15)."""
        if discipline_embeddings.size == 0:
            raise ValueError(
                "Невозможно вычислить вектор направления по пустому списку дисциплин."
            )

        mean_vector = np.mean(discipline_embeddings, axis=0)
        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            mean_vector = mean_vector / norm

        return np.asarray(mean_vector, dtype=np.float32)

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Вычисляет косинусное сходство между двумя векторами в диапазоне [0.0, 1.0]."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0

        cos_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        return max(0.0, min(1.0, cos_sim))