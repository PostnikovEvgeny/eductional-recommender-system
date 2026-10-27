import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.core.config import ROOT_DIR, settings

logger = logging.getLogger(__name__)


class FridaEmbedder:
    """
    Сервис управления локальной трансформерной моделью FRIDA и расчета эмбеддингов.
    Гарантирует однократное скачивание весов на диск и L2-нормализацию всех векторов.
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
        """
        Загружает модель из локальной папки ./models/frida.
        Если локальная папка пуста — скачивает модель один раз из Hugging Face и сохраняет на диск.
        """
        if self._model is not None:
            return self._model

        self.local_path.mkdir(parents=True, exist_ok=True)

        if self._is_model_downloaded():
            logger.info(
                "Загрузка локальной модели FRIDA с диска: %s (device=%s)",
                self.local_path,
                self.device,
            )
            self._model = SentenceTransformer(str(self.local_path), device=self.device)
        else:
            logger.info(
                "Локальная модель не найдена в %s. Скачивание '%s' из Hugging Face...",
                self.local_path,
                self.model_name,
            )
            model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Сохранение весов модели на диск в %s...", self.local_path)
            model.save(str(self.local_path))
            self._model = model
            logger.info("Модель FRIDA успешно сохранена локально и готова к работе.")

        return self._model

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        prefix: str = "search_document: ",
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Преобразует список текстов в матрицу нормализованных эмбеддингов (N x 1536).
        """
        if not texts:
            return np.empty((0, settings.embedding_dimension), dtype=np.float32)

        model = self.load_model()
        prefixed_texts = [f"{prefix}{t}" if prefix else t for t in texts]

        embeddings = model.encode(
            prefixed_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_documents(self, documents: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Кодирует тексты аннотаций дисциплин (префикс 'search_document: ')."""
        return self.encode_texts(documents, prefix="search_document: ", batch_size=batch_size)

    def encode_queries(self, queries: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Кодирует названия областей интереса / доменов (префикс 'search_query: ')."""
        return self.encode_texts(queries, prefix="search_query: ", batch_size=batch_size)

    def calculate_direction_centroid(
        self, discipline_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Вычисляет интегральный вектор направления подготовки V_D как нормализованное
        среднее арифметическое векторов профильных дисциплин (формула 15 диссертации).
        """
        if discipline_embeddings.size == 0:
            raise ValueError("Невозможно вычислить вектор направления по пустому списку дисциплин.")

        mean_vector = np.mean(discipline_embeddings, axis=0)
        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            mean_vector = mean_vector / norm

        return np.asarray(mean_vector, dtype=np.float32)

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Вычисляет косинусное сходство между двумя векторами в диапазоне [0.0, 1.0].
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0

        cos_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        return max(0.0, min(1.0, cos_sim))