import math
from dataclasses import dataclass

import numpy as np

from app.offline_pipeline.embedder import FridaEmbedder
from app.offline_pipeline.preprocessor import TextPreprocessor


@dataclass(frozen=True)
class ExtractedKeyword:
    """Извлеченный ключевой термин с комбинированной оценкой значимости и привязкой к источнику."""

    term: str
    semantic_score: float
    frequency: int
    embedding: np.ndarray
    source_discipline: str = ""
    is_anchor: bool = False


class FridaKeyBERTExtractor:
    """
    Автоматический экстрактор ключевых терминов по методу KeyBERT + MMR
    на базе локальной модели FRIDA с поддержкой глобального кэша эмбеддингов.
    """

    def __init__(
        self,
        preprocessor: TextPreprocessor,
        embedder: FridaEmbedder,
    ) -> None:
        self.preprocessor = preprocessor
        self.embedder = embedder

    @staticmethod
    def _lexical_overlap(term_a: str, term_b: str) -> float:
        """
        Вычисляет степень совпадения значимых корней слов между двумя фразами (от 0.0 до 1.0).
        """
        generic_connectors = {
            "разра", "прило", "систе", "данны", "техно", "метод", "инфор", "проек", "модел"
        }
        words_a = [
            w[:5] for w in term_a.lower().replace("-", " ").split() if len(w) >= 3
        ]
        words_b = [
            w[:5] for w in term_b.lower().replace("-", " ").split() if len(w) >= 3
        ]

        specific_a = {r for r in words_a if r not in generic_connectors} or set(words_a)
        specific_b = {r for r in words_b if r not in generic_connectors} or set(words_b)

        if not specific_a or not specific_b:
            return 0.0
        intersection = len(specific_a & specific_b)
        min_len = min(len(specific_a), len(specific_b))
        return intersection / min_len

    def extract_keywords_from_text(
        self,
        raw_text: str,
        *,
        discipline_title: str | None = None,
        excluded_terms: set[str] | None = None,
        precomputed_doc_vec: np.ndarray | None = None,
        candidate_vec_cache: dict[str, np.ndarray] | None = None,
        top_n: int = 5,
        min_semantic_score: float = 0.35,
        use_mmr: bool = True,
        diversity: float = 0.42,
    ) -> list[ExtractedKeyword]:
        """
        Извлекает наиболее релевантные смыслу дисциплины термины (KeyBERT + TF + MMR).
        Помечает термины из названия дисциплины флагом is_anchor=True.
        """
        dense_text = self.preprocessor.clean_for_dense(raw_text)
        if not dense_text:
            return []

        candidate_counts = self.preprocessor.extract_candidate_terms(raw_text)

        title_boost_terms: set[str] = set()
        if discipline_title:
            title_terms = self.preprocessor.extract_candidate_terms(discipline_title)
            for t_term, t_freq in title_terms.items():
                candidate_counts[t_term] += t_freq * 3
                title_boost_terms.add(t_term)

        if excluded_terms:
            lowered_excluded = {e.lower().strip() for e in excluded_terms}
            for term in list(candidate_counts.keys()):
                if term.lower() in lowered_excluded:
                    del candidate_counts[term]

        if not candidate_counts:
            return []

        candidate_terms = list(candidate_counts.keys())

        # 1. Используем предвычисленный вектор дисциплины (если передан), иначе считаем
        if precomputed_doc_vec is not None:
            doc_vec = precomputed_doc_vec
        else:
            doc_vec = self.embedder.encode_long_document(
                dense_text, prefix="categorize_topic: "
            )

        # 2. Берем векторы кандидатов из глобального батч-кэша (если передан)
        if candidate_vec_cache is not None:
            missing_terms = [t for t in candidate_terms if t not in candidate_vec_cache]
            if missing_terms:
                new_vecs = self.embedder.encode_texts(
                    missing_terms, prefix="categorize_topic: "
                )
                for t, v in zip(missing_terms, new_vecs, strict=True):
                    candidate_vec_cache[t] = v
            cand_vecs = np.array(
                [candidate_vec_cache[t] for t in candidate_terms], dtype=np.float32
            )
        else:
            cand_vecs = self.embedder.encode_texts(
                candidate_terms, prefix="categorize_topic: "
            )

        # 3. Косинусная близость + бонус за частоту и присутствие в заголовке дисциплины
        raw_cosines = np.dot(cand_vecs, doc_vec)
        weighted_scores: list[float] = []
        for idx, term in enumerate(candidate_terms):
            cos_val = max(0.0, float(raw_cosines[idx]))
            freq = candidate_counts[term]
            tf_boost = 1.0 + 0.10 * math.log(freq) if freq > 1 else 1.0
            title_multiplier = 1.08 if term in title_boost_terms else 1.0
            weighted_scores.append(min(1.0, cos_val * tf_boost * title_multiplier))

        valid_indices = [
            idx
            for idx, score in enumerate(weighted_scores)
            if score >= min_semantic_score
        ]
        if not valid_indices:
            valid_indices = list(range(len(candidate_terms)))

        source_name = discipline_title or ""

        if not use_mmr or len(valid_indices) <= top_n:
            sorted_idx = sorted(
                valid_indices, key=lambda idx: weighted_scores[idx], reverse=True
            )[:top_n]
            return [
                ExtractedKeyword(
                    term=candidate_terms[idx],
                    semantic_score=round(weighted_scores[idx], 4),
                    frequency=candidate_counts[candidate_terms[idx]],
                    embedding=cand_vecs[idx],
                    source_discipline=source_name,
                    is_anchor=(candidate_terms[idx] in title_boost_terms),
                )
                for idx in sorted_idx
            ]

        selected_indices: list[int] = []
        remaining_indices = list(valid_indices)

        best_first = max(remaining_indices, key=lambda idx: weighted_scores[idx])
        selected_indices.append(best_first)
        remaining_indices.remove(best_first)

        lambda_param = 1.0 - diversity

        while len(selected_indices) < top_n and remaining_indices:
            best_mmr_score = -float("inf")
            best_idx = -1

            for idx in remaining_indices:
                relevance = weighted_scores[idx]
                redundancy = max(
                    max(
                        float(np.dot(cand_vecs[idx], cand_vecs[sel_idx])),
                        self._lexical_overlap(
                            candidate_terms[idx], candidate_terms[sel_idx]
                        ),
                    )
                    for sel_idx in selected_indices
                )
                mmr_score = lambda_param * relevance - (1.0 - lambda_param) * redundancy

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            if best_idx == -1:
                break

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        selected_indices.sort(key=lambda idx: weighted_scores[idx], reverse=True)

        return [
            ExtractedKeyword(
                term=candidate_terms[idx],
                semantic_score=round(weighted_scores[idx], 4),
                frequency=candidate_counts[candidate_terms[idx]],
                embedding=cand_vecs[idx],
                source_discipline=source_name,
                is_anchor=(candidate_terms[idx] in title_boost_terms),
            )
            for idx in selected_indices
        ]