from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from app.offline_pipeline.keybert_extractor import ExtractedKeyword, FridaKeyBERTExtractor

COMPRESSION_LEVELS: tuple[int, ...] = (1, 3, 5)


@dataclass(frozen=True)
class ClusteredDomain:
    """
    Сформированный предметный домен (область интереса абитуриента)
    для заданного уровня семантического сжатия (1, 3 или 5).
    """

    name: str                      # Человекочитаемое название кнопки (с заглавной буквы)
    compression_level: int         # Уровень сжатия: 1, 3 или 5
    centroid_embedding: np.ndarray # Нормализованный вектор домена V_domain (1536,)
    member_terms: list[str]        # Список исходных терминов, вошедших в этот кластер
    cohesion_score: float          # Внутренняя семантическая плотность кластера [0, 1]


class HierarchicalDomainClusterer:
    """
    Модуль иерархического агломеративного семантического сжатия (HAC).
    Использует метод минимизации дисперсии Уорда (Ward linkage) на единичной гиперсфере
    для получения сбалансированных кластеров и алгоритм композитного лейблирования.
    """

    @staticmethod
    def _format_label(term: str) -> str:
        """Приводит первую букву названия интереса к верхнему регистру для UI."""
        cleaned = term.strip()
        if not cleaned:
            return ""
        return cleaned[0].upper() + cleaned[1:]

    @staticmethod
    def _calculate_weighted_centroid(
        members: list[ExtractedKeyword],
    ) -> np.ndarray:
        """
        Вычисляет L2-нормализованный взвешенный центроид кластера V_domain.
        """
        vectors = np.array([m.embedding for m in members], dtype=np.float32)
        weights = np.array(
            [max(0.01, m.semantic_score) for m in members], dtype=np.float32
        )

        weighted_sum = np.average(vectors, axis=0, weights=weights)
        norm = np.linalg.norm(weighted_sum)
        if norm > 0:
            weighted_sum = weighted_sum / norm

        return np.asarray(weighted_sum, dtype=np.float32)

    def _rank_candidates_in_cluster(
        self,
        members: list[ExtractedKeyword],
        centroid: np.ndarray,
    ) -> list[ExtractedKeyword]:
        """Ранжирует термины внутри кластера по их центральной представительности."""

        def _score_item(item: ExtractedKeyword) -> float:
            centrality = float(np.dot(item.embedding, centroid))
            word_count = len(item.term.split())
            if word_count in (2, 3):
                length_multiplier = 1.15
            elif word_count == 1:
                length_multiplier = 0.95
            else:
                length_multiplier = 0.85

            freq_bonus = 1.0 + 0.05 * min(item.frequency, 5)
            return (0.60 * centrality + 0.40 * item.semantic_score) * length_multiplier * freq_bonus

        return sorted(members, key=_score_item, reverse=True)

    def _select_best_cluster_label(
        self,
        members: list[ExtractedKeyword],
        centroid: np.ndarray,
        *,
        allow_composite: bool = False,
        n_subclusters: int = 2,
    ) -> str:
        """
        Выбирает наиболее понятное название кластера.
        При allow_composite=True разбивает группу на подкластеры по Уорду,
        выбирает 2 крупнейших по массе (числу терминов) полюса и соединяет их союзом 'и'.
        """
        ranked = self._rank_candidates_in_cluster(members, centroid)
        primary = ranked[0]

        if not allow_composite or len(members) < 6:
            return self._format_label(primary.term)

        sub_matrix = np.array([m.embedding for m in members], dtype=np.float64)
        sub_dist = pdist(sub_matrix, metric="euclidean")
        sub_link = linkage(sub_dist, method="ward")
        sub_labels = fcluster(sub_link, t=n_subclusters, criterion="maxclust")

        subgroups: dict[int, list[ExtractedKeyword]] = defaultdict(list)
        for m, lbl in zip(members, sub_labels, strict=True):
            subgroups[int(lbl)].append(m)

        # Сортируем подгруппы по суммарному весу (массе) входящих в них терминов,
        # чтобы мелкие побочные темы не вытесняли главные ядра программы
        sorted_groups = sorted(
            subgroups.values(),
            key=lambda g: (len(g), sum(item.semantic_score for item in g)),
            reverse=True,
        )

        if len(sorted_groups) < 2:
            return self._format_label(primary.term)

        top_group_1 = sorted_groups[0]
        top_group_2 = sorted_groups[1]

        c1 = self._calculate_weighted_centroid(top_group_1)
        c2 = self._calculate_weighted_centroid(top_group_2)
        leader_1 = self._rank_candidates_in_cluster(top_group_1, c1)[0]
        leader_2 = self._rank_candidates_in_cluster(top_group_2, c2)[0]

        overlap = FridaKeyBERTExtractor._lexical_overlap(leader_1.term, leader_2.term)
        if overlap < 0.5:
            composite_name = f"{leader_1.term} и {leader_2.term}"
            if len(composite_name.split()) <= 7:
                return self._format_label(composite_name)

        return self._format_label(primary.term)

    def cluster_for_level(
        self,
        keywords: list[ExtractedKeyword],
        target_k: int,
        *,
        macro_direction_title: str | None = None,
    ) -> list[ClusteredDomain]:
        """
        Выполняет срез дендрограммы HAC на заданное число кластеров target_k (1, 3 или 5).
        """
        if target_k <= 0:
            raise ValueError("Целевое количество кластеров target_k должно быть положительным.")

        if not keywords:
            return []

        n_items = len(keywords)

        # Краевой случай: если терминов меньше или равно target_k
        if n_items == 1 or n_items <= target_k:
            results: list[ClusteredDomain] = []
            for kw in keywords[:target_k]:
                norm_vec = kw.embedding / max(float(np.linalg.norm(kw.embedding)), 1e-9)
                results.append(
                    ClusteredDomain(
                        name=self._format_label(kw.term),
                        compression_level=target_k,
                        centroid_embedding=np.asarray(norm_vec, dtype=np.float32),
                        member_terms=[kw.term],
                        cohesion_score=1.0,
                    )
                )
            return results

        # Уровень 1 (макро-концепт всего направления)
        if target_k == 1:
            centroid = self._calculate_weighted_centroid(keywords)
            cohesion = float(
                np.mean([max(0.0, float(np.dot(m.embedding, centroid))) for m in keywords])
            )
            if macro_direction_title:
                label = self._format_label(macro_direction_title)
            else:
                label = self._select_best_cluster_label(
                    keywords, centroid, allow_composite=True, n_subclusters=3
                )
            return [
                ClusteredDomain(
                    name=label,
                    compression_level=1,
                    centroid_embedding=centroid,
                    member_terms=[m.term for m in keywords],
                    cohesion_score=round(cohesion, 4),
                )
            ]

        # 1. Собираем матрицу L2-нормализованных векторов (N x 1536)
        embeddings_matrix = np.array([kw.embedding for kw in keywords], dtype=np.float64)

        # 2. Используем метод Уорда (ward) на нормализованных векторах (сферический Ward):
        # он минимизирует дисперсию внутри кластеров и исключает появление кластеров-одиночек из 1 слова!
        condensed_distances = pdist(embeddings_matrix, metric="euclidean")
        linkage_matrix = linkage(condensed_distances, method="ward")

        # 3. Горизонтальный срез дерева на ровно target_k кластеров
        cluster_labels = fcluster(linkage_matrix, t=target_k, criterion="maxclust")

        grouped_members: dict[int, list[ExtractedKeyword]] = defaultdict(list)
        for kw, cluster_id in zip(keywords, cluster_labels, strict=True):
            grouped_members[int(cluster_id)].append(kw)

        # 4. Формируем итоговые домены для каждого кластера
        # Для уровня 3 (мезо-уровень) включаем составные названия из 2 подтем
        allow_composite = target_k <= 3
        clustered_domains: list[ClusteredDomain] = []

        for members in grouped_members.values():
            centroid = self._calculate_weighted_centroid(members)
            label = self._select_best_cluster_label(
                members, centroid, allow_composite=allow_composite
            )
            cohesion = float(
                np.mean([max(0.0, float(np.dot(m.embedding, centroid))) for m in members])
            )
            clustered_domains.append(
                ClusteredDomain(
                    name=label,
                    compression_level=target_k,
                    centroid_embedding=centroid,
                    member_terms=[m.term for m in members],
                    cohesion_score=round(cohesion, 4),
                )
            )

        clustered_domains.sort(
            key=lambda d: len(d.member_terms),
            reverse=True,
        )
        return clustered_domains

    def build_multiscale_hierarchy(
        self,
        keywords: list[ExtractedKeyword],
        levels: tuple[int, ...] = COMPRESSION_LEVELS,
        *,
        macro_direction_title: str | None = None,
    ) -> dict[int, list[ClusteredDomain]]:
        """
        Строит полную трехуровневую иерархию сжатия информации (уровни 1, 3 и 5).
        """
        hierarchy: dict[int, list[ClusteredDomain]] = {}
        for k in levels:
            hierarchy[k] = self.cluster_for_level(
                keywords,
                target_k=k,
                macro_direction_title=macro_direction_title if k == 1 else None,
            )
        return hierarchy