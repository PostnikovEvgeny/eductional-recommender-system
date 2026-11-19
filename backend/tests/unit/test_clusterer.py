import math
import numpy as np

from app.offline_pipeline.clusterer import ClusteringConfig, HierarchicalDomainClusterer
from app.offline_pipeline.keybert_extractor import ExtractedKeyword


def _make_keyword(
    term: str,
    base_vec: np.ndarray,
    score: float = 0.9,
    freq: int = 2,
    source_discipline: str = "",
    is_anchor: bool = False,
) -> ExtractedKeyword:
    """Вспомогательная фабрика тестовых ключевых слов с нормализованным вектором."""
    norm = np.linalg.norm(base_vec)
    unit_vec = (base_vec / norm).astype(np.float32)
    return ExtractedKeyword(
        term=term,
        semantic_score=score,
        frequency=freq,
        embedding=unit_vec,
        source_discipline=source_discipline,
        is_anchor=is_anchor,
    )


class TestHierarchicalDomainClusterer:
    """Тестирование иерархической кластеризации HAC для сжатия 1-3-5 и HITL-карточек."""

    def test_multiscale_hierarchy_returns_exact_1_3_5_counts(self) -> None:
        """Проверка формирования ровно 1, 3 и 5 кластеров из пула 10 терминов."""
        rng = np.random.default_rng(seed=42)
        keywords = [
            _make_keyword(f"термин направление {i}", rng.random(1536))
            for i in range(10)
        ]

        clusterer = HierarchicalDomainClusterer()
        hierarchy = clusterer.build_multiscale_hierarchy(keywords, levels=(1, 3, 5))

        assert len(hierarchy[1]) == 1
        assert len(hierarchy[3]) == 3
        assert len(hierarchy[5]) == 5

        total_members_in_5 = sum(len(c.member_terms) for c in hierarchy[5])
        assert total_members_in_5 == 10

        for level_domains in hierarchy.values():
            for domain in level_domains:
                norm = float(np.linalg.norm(domain.centroid_embedding))
                assert math.isclose(norm, 1.0, abs_tol=1e-5)
                assert domain.name[0].isupper()

    def test_semantically_close_terms_merge_into_same_cluster(self) -> None:
        """Проверка того, что близкие по вектору термины объединяются в один кластер."""
        axis_ml = np.zeros(1536, dtype=np.float32)
        axis_ml[0] = 1.0

        axis_db = np.zeros(1536, dtype=np.float32)
        axis_db[1] = 1.0

        axis_web = np.zeros(1536, dtype=np.float32)
        axis_web[2] = 1.0

        keywords = [
            _make_keyword("интеллектуальный анализ данных", axis_ml + 0.01, score=0.95),
            _make_keyword("классификация", axis_ml + 0.02, score=0.85),
            _make_keyword("реляционная база данных", axis_db + 0.01, score=0.92),
            _make_keyword("проектирование хранилищ", axis_db + 0.02, score=0.80),
            _make_keyword("клиент-серверное приложение", axis_web + 0.01, score=0.90),
            _make_keyword("веб-разработка", axis_web + 0.02, score=0.88),
        ]

        clusterer = HierarchicalDomainClusterer()
        clusters_3 = clusterer.cluster_for_level(keywords, target_k=3)

        assert len(clusters_3) == 3
        member_sets = [set(c.member_terms) for c in clusters_3]

        assert {"интеллектуальный анализ данных", "классификация"} in member_sets
        assert {"реляционная база данных", "проектирование хранилищ"} in member_sets
        assert {"клиент-серверное приложение", "веб-разработка"} in member_sets

    def test_hitl_cluster_card_populates_alternatives_and_respects_config(self) -> None:
        """Проверка генерации альтернативных названий и дисциплин для панели инженера."""
        axis_road = np.zeros(1536, dtype=np.float32)
        axis_road[0] = 1.0

        keywords = [
            _make_keyword(
                "дорожная одежда",
                axis_road + 0.01,
                score=0.94,
                freq=3,
                source_discipline="Технология и организация строительства",
                is_anchor=False,
            ),
            _make_keyword(
                "дорожное строительство",
                axis_road + 0.02,
                score=0.92,
                freq=3,
                source_discipline="Технология и организация строительства",
                is_anchor=True,
            ),
            _make_keyword(
                "асфальтобетонная смесь",
                axis_road + 0.03,
                score=0.88,
                freq=2,
                source_discipline="Дорожно-строительные материалы",
                is_anchor=False,
            ),
        ]

        cfg = ClusteringConfig(anchor_weight=1.40, max_alternative_labels=4)
        clusterer = HierarchicalDomainClusterer()
        domains = clusterer.cluster_for_level(keywords, target_k=1, config=cfg)

        assert len(domains) == 1
        dom = domains[0]
        # При anchor_weight=1.40 якорный термин "Дорожное строительство" побеждает
        assert dom.name == "Дорожное строительство"
        assert "Дорожная одежда" in dom.alternative_labels
        assert "Технология и организация строительства" in dom.source_disciplines
        assert len(dom.alternative_labels) <= 4