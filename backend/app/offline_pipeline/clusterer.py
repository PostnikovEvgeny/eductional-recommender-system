from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from app.offline_pipeline.keybert_extractor import ExtractedKeyword, FridaKeyBERTExtractor

COMPRESSION_LEVELS: tuple[int, ...] = (1, 3, 5)


@dataclass(frozen=True)
class ClusteringConfig:
    """
    Настраиваемые параметры алгоритма иерархического сжатия HAC (1-3-5).
    Используются в панели инженера-методиста для мгновенного пересчета кластеров
    по закэшированным эмбеддингам без повторного запуска модели FRIDA.
    """

    anchor_weight: float = 1.28          # Приоритет якорных терминов (из заголовков дисциплин)
    rebalance_clusters: bool = True      # Балансировка размеров кластеров (убирает выбросы)
    allow_composite_level_3: bool = True # Составные названия "А и Б" на уровне 3
    broad_term_penalty: float = 0.72     # Штраф для слишком общих терминов-гиперонимов
    max_alternative_labels: int = 5      # Количество альтернативных вариантов названия для UI


@dataclass(frozen=True)
class ClusteredDomain:
    """
    Сформированный предметный домен (карточка кластера / кнопка интереса)
    для заданного уровня семантического сжатия (1, 3 или 5).
    """

    name: str                                            # Рекомендованное алгоритмом название кнопки
    compression_level: int                               # Уровень сжатия: 1, 3 или 5
    centroid_embedding: np.ndarray                       # Нормализованный вектор домена V_domain (1536,)
    member_terms: list[str]                              # Список терминов, вошедших в кластер
    cohesion_score: float                                # Внутренняя семантическая плотность [0, 1]
    alternative_labels: list[str] = field(default_factory=list)  # Топ-N альтернатив для выбора инженером
    source_disciplines: list[str] = field(default_factory=list)  # Главные дисциплины, сформировавшие кластер


class HierarchicalDomainClusterer:
    """
    Модуль иерархического агломеративного семантического сжатия (HAC).
    Использует метод Уорда (Ward linkage) с настраиваемой конфигурацией ClusteringConfig
    и генерацией карточек альтернативных названий для HITL-верификации.
    """

    GENERIC_BROAD_TERMS: set[str] = {
        "информационная система",
        "программная система",
        "автоматизированная система",
        "вычислительная машина",
        "программный продукт",
        "строительный материал",
        "программный интерфейс",
    }

    def __init__(self, default_config: ClusteringConfig | None = None) -> None:
        self.default_config = default_config or ClusteringConfig()

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
        """Вычисляет L2-нормализованный взвешенный центроид кластера V_domain."""
        vectors = np.array([m.embedding for m in members], dtype=np.float32)
        weights = np.array(
            [max(0.01, m.semantic_score) for m in members], dtype=np.float32
        )

        weighted_sum = np.average(vectors, axis=0, weights=weights)
        norm = np.linalg.norm(weighted_sum)
        if norm > 0:
            weighted_sum = weighted_sum / norm

        return np.asarray(weighted_sum, dtype=np.float32)

    @staticmethod
    def _extract_top_disciplines(members: list[ExtractedKeyword]) -> list[str]:
        """Возвращает упорядоченный по числу терминов список дисциплин кластера."""
        counts = Counter(
            m.source_discipline.strip()
            for m in members
            if m.source_discipline and m.source_discipline.strip()
        )
        return [disc for disc, _ in counts.most_common(5)]

    def _rank_candidates_in_cluster(
        self,
        members: list[ExtractedKeyword],
        centroid: np.ndarray,
        config: ClusteringConfig,
        *,
        penalize_broad_terms: bool = False,
    ) -> list[ExtractedKeyword]:
        """Ранжирует термины внутри кластера для выбора основного и альтернативных названий."""
        eligible_pool = [m for m in members if m.is_anchor or m.frequency >= 2]
        pool = eligible_pool if eligible_pool else members

        def _score_item(item: ExtractedKeyword) -> float:
            centrality = float(np.dot(item.embedding, centroid))
            word_count = len(item.term.split())
            if word_count in (2, 3):
                length_multiplier = 1.16
            elif word_count == 1:
                length_multiplier = 0.90
            else:
                length_multiplier = 0.85

            anchor_bonus = config.anchor_weight if item.is_anchor else 1.00

            broad_penalty = (
                config.broad_term_penalty
                if (
                    penalize_broad_terms
                    and item.term.lower() in self.GENERIC_BROAD_TERMS
                )
                else 1.0
            )
            freq_bonus = 1.0 + 0.06 * min(item.frequency, 6)
            return (
                (0.55 * centrality + 0.45 * item.semantic_score)
                * length_multiplier
                * anchor_bonus
                * freq_bonus
                * broad_penalty
            )

        return sorted(pool, key=_score_item, reverse=True)

    def _build_alternative_labels(
        self,
        primary_label: str,
        ranked_members: list[ExtractedKeyword],
        source_disciplines: list[str],
        max_alternatives: int,
    ) -> list[str]:
        """
        Формирует список лучших альтернативных формулировок для карточки кластера в панели инженера:
        объединяет топовые термины кластера и названия главных дисциплин без дубликатов.
        """
        alternatives: list[str] = []
        seen_lower: set[str] = {primary_label.lower()}

        for item in ranked_members:
            formatted = self._format_label(item.term)
            if formatted.lower() not in seen_lower:
                alternatives.append(formatted)
                seen_lower.add(formatted.lower())
            if len(alternatives) >= max_alternatives:
                return alternatives

        # Дополняем названиями ведущих дисциплин кластера (часто они звучат яснее узких терминов)
        for disc in source_disciplines:
            formatted_disc = self._format_label(disc)
            if formatted_disc.lower() not in seen_lower:
                alternatives.append(formatted_disc)
                seen_lower.add(formatted_disc.lower())
            if len(alternatives) >= max_alternatives:
                break

        return alternatives

    def _select_best_cluster_label(
        self,
        members: list[ExtractedKeyword],
        centroid: np.ndarray,
        config: ClusteringConfig,
        *,
        allow_composite: bool = False,
        n_subclusters: int = 2,
        penalize_broad_terms: bool = False,
    ) -> tuple[str, list[ExtractedKeyword]]:
        """
        Выбирает основное название кластера и возвращает отсортированный пул кандидатов.
        При allow_composite=True выбирает 2 главных полюса кластера из разных дисциплин.
        """
        ranked = self._rank_candidates_in_cluster(
            members, centroid, config, penalize_broad_terms=penalize_broad_terms
        )
        primary = ranked[0]

        if not allow_composite or len(members) < 5:
            return self._format_label(primary.term), ranked

        sub_matrix = np.array([m.embedding for m in members], dtype=np.float64)
        sub_dist = pdist(sub_matrix, metric="euclidean")
        sub_link = linkage(sub_dist, method="ward")
        sub_labels = fcluster(sub_link, t=n_subclusters, criterion="maxclust")

        subgroups: dict[int, list[ExtractedKeyword]] = defaultdict(list)
        for m, lbl in zip(members, sub_labels, strict=True):
            subgroups[int(lbl)].append(m)

        sorted_groups = sorted(
            subgroups.values(),
            key=lambda g: (len(g), sum(item.semantic_score for item in g)),
            reverse=True,
        )

        if len(sorted_groups) < 2:
            return self._format_label(primary.term), ranked

        top_group_1 = sorted_groups[0]
        c1 = self._calculate_weighted_centroid(top_group_1)
        leader_1 = self._rank_candidates_in_cluster(
            top_group_1, c1, config, penalize_broad_terms=penalize_broad_terms
        )[0]

        for second_group in sorted_groups[1:]:
            c2 = self._calculate_weighted_centroid(second_group)
            candidates_2 = self._rank_candidates_in_cluster(
                second_group, c2, config, penalize_broad_terms=penalize_broad_terms
            )
            for cand_2 in candidates_2:
                if not (cand_2.is_anchor or cand_2.frequency >= 2):
                    continue
                same_disc = (
                    bool(leader_1.source_discipline)
                    and bool(cand_2.source_discipline)
                    and leader_1.source_discipline == cand_2.source_discipline
                )
                if same_disc:
                    continue
                overlap = FridaKeyBERTExtractor._lexical_overlap(
                    leader_1.term, cand_2.term
                )
                if overlap == 0.0:
                    composite_name = f"{leader_1.term} и {cand_2.term}"
                    if len(composite_name.split()) <= 7:
                        return self._format_label(composite_name), ranked

        return self._format_label(primary.term), ranked

    def _rebalance_clusters(
        self,
        clusters: list[list[ExtractedKeyword]],
        target_k: int,
        total_items: int,
    ) -> list[list[ExtractedKeyword]]:
        """Устраняет экстремальный дисбаланс размеров кластеров."""
        if len(clusters) != target_k or total_items < 12:
            return clusters

        min_allowed_size = max(3, int(total_items * 0.10))
        max_allowed_size = int(total_items * 0.45)

        balanced = [list(c) for c in clusters]

        for _ in range(3):
            balanced.sort(key=len, reverse=True)
            largest = balanced[0]
            smallest = balanced[-1]

            if len(largest) <= max_allowed_size and len(smallest) > min_allowed_size:
                break

            if len(smallest) <= min_allowed_size and len(largest) >= 8:
                small_centroid = self._calculate_weighted_centroid(smallest)
                best_merge_idx = 0
                best_sim = -1.0
                for idx in range(len(balanced) - 1):
                    cand_centroid = self._calculate_weighted_centroid(balanced[idx])
                    sim = float(np.dot(small_centroid, cand_centroid))
                    if sim > best_sim:
                        best_sim = sim
                        best_merge_idx = idx

                balanced[best_merge_idx].extend(smallest)
                balanced.pop()

                balanced.sort(key=len, reverse=True)
                largest_now = balanced[0]
                sub_matrix = np.array(
                    [m.embedding for m in largest_now], dtype=np.float64
                )
                sub_dist = pdist(sub_matrix, metric="euclidean")
                sub_link = linkage(sub_dist, method="ward")
                sub_labels = fcluster(sub_link, t=2, criterion="maxclust")

                part_1 = [
                    m
                    for m, lbl in zip(largest_now, sub_labels, strict=True)
                    if int(lbl) == 1
                ]
                part_2 = [
                    m
                    for m, lbl in zip(largest_now, sub_labels, strict=True)
                    if int(lbl) == 2
                ]

                if part_1 and part_2:
                    balanced[0] = part_1
                    balanced.append(part_2)
                else:
                    break
            else:
                break

        return balanced

    def cluster_for_level(
        self,
        keywords: list[ExtractedKeyword],
        target_k: int,
        *,
        macro_direction_title: str | None = None,
        config: ClusteringConfig | None = None,
    ) -> list[ClusteredDomain]:
        """Выполняет срез дендрограммы HAC на заданное число кластеров target_k (1, 3 или 5)."""
        if target_k <= 0:
            raise ValueError(
                "Целевое количество кластеров target_k должно быть положительным."
            )

        if not keywords:
            return []

        cfg = config or self.default_config
        n_items = len(keywords)

        if n_items == 1 or n_items <= target_k:
            results: list[ClusteredDomain] = []
            for kw in keywords[:target_k]:
                norm_vec = kw.embedding / max(float(np.linalg.norm(kw.embedding)), 1e-9)
                disc_list = [kw.source_discipline] if kw.source_discipline else []
                results.append(
                    ClusteredDomain(
                        name=self._format_label(kw.term),
                        compression_level=target_k,
                        centroid_embedding=np.asarray(norm_vec, dtype=np.float32),
                        member_terms=[kw.term],
                        cohesion_score=1.0,
                        alternative_labels=[],
                        source_disciplines=disc_list,
                    )
                )
            return results

        if target_k == 1:
            centroid = self._calculate_weighted_centroid(keywords)
            cohesion = float(
                np.mean(
                    [max(0.0, float(np.dot(m.embedding, centroid))) for m in keywords]
                )
            )
            source_discs = self._extract_top_disciplines(keywords)
            auto_label, ranked = self._select_best_cluster_label(
                keywords,
                centroid,
                cfg,
                allow_composite=True,
                n_subclusters=3,
                penalize_broad_terms=True,
            )
            final_label = (
                self._format_label(macro_direction_title)
                if macro_direction_title
                else auto_label
            )
            alts = self._build_alternative_labels(
                final_label, ranked, source_discs, cfg.max_alternative_labels
            )
            if macro_direction_title and auto_label.lower() != final_label.lower():
                alts = [auto_label, *[a for a in alts if a.lower() != auto_label.lower()]][
                    : cfg.max_alternative_labels
                ]

            return [
                ClusteredDomain(
                    name=final_label,
                    compression_level=1,
                    centroid_embedding=centroid,
                    member_terms=[m.term for m in keywords],
                    cohesion_score=round(cohesion, 4),
                    alternative_labels=alts,
                    source_disciplines=source_discs,
                )
            ]

        embeddings_matrix = np.array([kw.embedding for kw in keywords], dtype=np.float64)
        condensed_distances = pdist(embeddings_matrix, metric="euclidean")
        linkage_matrix = linkage(condensed_distances, method="ward")
        cluster_labels = fcluster(linkage_matrix, t=target_k, criterion="maxclust")

        grouped_members: dict[int, list[ExtractedKeyword]] = defaultdict(list)
        for kw, cluster_id in zip(keywords, cluster_labels, strict=True):
            grouped_members[int(cluster_id)].append(kw)

        raw_clusters = list(grouped_members.values())
        final_clusters = (
            self._rebalance_clusters(raw_clusters, target_k=target_k, total_items=n_items)
            if cfg.rebalance_clusters
            else raw_clusters
        )

        allow_composite = target_k <= 3 and cfg.allow_composite_level_3
        clustered_domains: list[ClusteredDomain] = []

        for members in final_clusters:
            centroid = self._calculate_weighted_centroid(members)
            source_discs = self._extract_top_disciplines(members)
            label, ranked = self._select_best_cluster_label(
                members,
                centroid,
                cfg,
                allow_composite=allow_composite,
                penalize_broad_terms=True,
            )
            alternatives = self._build_alternative_labels(
                label, ranked, source_discs, cfg.max_alternative_labels
            )
            cohesion = float(
                np.mean(
                    [max(0.0, float(np.dot(m.embedding, centroid))) for m in members]
                )
            )
            clustered_domains.append(
                ClusteredDomain(
                    name=label,
                    compression_level=target_k,
                    centroid_embedding=centroid,
                    member_terms=[m.term for m in members],
                    cohesion_score=round(cohesion, 4),
                    alternative_labels=alternatives,
                    source_disciplines=source_discs,
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
        config: ClusteringConfig | None = None,
    ) -> dict[int, list[ClusteredDomain]]:
        """Строит полную трехуровневую иерархию сжатия информации (уровни 1, 3 и 5)."""
        hierarchy: dict[int, list[ClusteredDomain]] = {}
        for k in levels:
            hierarchy[k] = self.cluster_for_level(
                keywords,
                target_k=k,
                macro_direction_title=macro_direction_title if k == 1 else None,
                config=config,
            )
        return hierarchy