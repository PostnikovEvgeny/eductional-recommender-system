import math
import pytest

from app.core.exceptions import (
    DimensionMismatchError,
    InvalidAffiliationValueError,
    InvalidDomainsCountError,
)
from app.domain import (
    calculate_core_score,
    calculate_percentile_ranks,
    calculate_roc_weights,
    calculate_wsm_score,
    calculate_wsm_with_contributions,
    map_domains_to_roc_weights,
)


class TestROCWeights:
    """Тестирование расчета весовых коэффициентов методом Rank Order Centroid (ROC)."""

    def test_roc_weights_for_5_domains_matches_thesis_example(self) -> None:
        """Проверка эталонного примера из главы 1.3.3 и 2.1 диссертации для k = 5."""
        weights = calculate_roc_weights(5)

        assert len(weights) == 5
        assert math.isclose(sum(weights), 1.0, abs_tol=1e-9)
        # Сверка с расчетными значениями из текста диссертации:
        # w1 ≈ 0.4567, w2 ≈ 0.2567, w3 ≈ 0.1567, w4 ≈ 0.0900, w5 ≈ 0.0400
        assert math.isclose(weights[0], 0.456666, abs_tol=1e-4)
        assert math.isclose(weights[1], 0.256666, abs_tol=1e-4)
        assert math.isclose(weights[2], 0.156666, abs_tol=1e-4)
        assert math.isclose(weights[3], 0.090000, abs_tol=1e-4)
        assert math.isclose(weights[4], 0.040000, abs_tol=1e-4)

    def test_roc_weights_for_3_domains(self) -> None:
        """Проверка расчета весов ROC для k = 3 (минимальный порог выбора)."""
        weights = calculate_roc_weights(3)

        assert len(weights) == 3
        assert math.isclose(sum(weights), 1.0, abs_tol=1e-9)
        assert math.isclose(weights[0], 11 / 18, abs_tol=1e-6)  # ~0.6111
        assert math.isclose(weights[1], 5 / 18, abs_tol=1e-6)   # ~0.2778
        assert math.isclose(weights[2], 2 / 18, abs_tol=1e-6)   # ~0.1111

    @pytest.mark.parametrize("invalid_k", [0, 1, 2, 6, 10, -3])
    def test_roc_raises_error_when_count_out_of_bounds(self, invalid_k: int) -> None:
        """Проверка блокировки выбора менее 3 или более 5 интересов."""
        with pytest.raises(InvalidDomainsCountError):
            calculate_roc_weights(invalid_k)

    def test_map_domains_to_roc_weights(self) -> None:
        """Проверка привязки ID доменов к их весам."""
        domain_ids = [10, 4, 22]
        mapped = map_domains_to_roc_weights(domain_ids)

        assert list(mapped.keys()) == [10, 4, 22]
        assert math.isclose(mapped[10], 11 / 18, abs_tol=1e-6)


class TestWSMModel:
    """Тестирование метода взвешенной суммы (Weighted Sum Model)."""

    def test_wsm_matches_thesis_applied_math_example(self) -> None:
        """
        Проверка контрольного примера из п. 2.1 диссертации:
        Веса: Математика=0.6, IT=0.3, Экология=0.1
        Направление 1 ('Прикладная математика'): mu = [0.9, 0.3, 0.1] -> Score = 0.64
        Направление 2 ('ПМИ'): mu = [0.5, 0.7, 0.1] -> Score = 0.52
        """
        weights = [0.6, 0.3, 0.1]

        score_pm = calculate_wsm_score(weights, [0.9, 0.3, 0.1])
        score_pmi = calculate_wsm_score(weights, [0.5, 0.7, 0.1])

        assert math.isclose(score_pm, 0.64, abs_tol=1e-6)
        assert math.isclose(score_pmi, 0.52, abs_tol=1e-6)

    def test_wsm_with_contributions_decomposition(self) -> None:
        """Проверка детализации вклада каждого интереса в итоговую оценку."""
        domain_weights = {1: 0.6, 2: 0.3, 3: 0.1}
        domain_affiliations = {1: 0.9, 2: 0.3, 3: 0.1}

        total_score, contributions = calculate_wsm_with_contributions(
            domain_weights, domain_affiliations
        )

        assert math.isclose(total_score, 0.64, abs_tol=1e-6)
        assert math.isclose(contributions[1], 0.54, abs_tol=1e-6)
        assert math.isclose(contributions[2], 0.09, abs_tol=1e-6)
        assert math.isclose(contributions[3], 0.01, abs_tol=1e-6)

    def test_wsm_dimension_mismatch_raises_error(self) -> None:
        """Проверка выброса исключения при разной длине списков весов и оценок."""
        with pytest.raises(DimensionMismatchError):
            calculate_wsm_score([0.6, 0.4], [0.9, 0.5, 0.1])

    def test_wsm_invalid_affiliation_raises_error(self) -> None:
        """Проверка выброса исключения, если близость больше 1.0 или меньше 0.0."""
        with pytest.raises(InvalidAffiliationValueError):
            calculate_wsm_score([0.6, 0.4], [1.25, 0.5])


class TestPercentileAndCoreAggregation:
    """Тестирование процентильного ранга и агрегации оценок ядер направлений."""

    def test_percentile_rank_matches_thesis_table_1(self) -> None:
        """
        Проверка контрольного примера из Таблицы 1 диссертации (7 направлений инженерного ядра):
        Для направления 'Наноматериалы' (score = 0.77): L = 4, E = 1, N = 7 -> PR = 64.29%
        """
        engineering_scores = {
            1: 0.91,  # Мехатроника и робототехника
            2: 0.84,  # Машиностроение
            3: 0.77,  # Наноматериалы
            4: 0.72,  # Автоматизация производств
            5: 0.69,  # Приборостроение
            6: 0.58,  # Материаловедение
            7: 0.41,  # Техносферная безопасность
        }

        pr_ranks = calculate_percentile_ranks(engineering_scores)

        # Для Наноматериалов (id=3): ((4 + 0.5 * 1) / 7) * 100 = 64.2857... -> 64.29
        assert math.isclose(pr_ranks[3], 64.29, abs_tol=1e-2)
        # Для лидера (id=1): ((6 + 0.5 * 1) / 7) * 100 = 92.86
        assert math.isclose(pr_ranks[1], 92.86, abs_tol=1e-2)

    def test_core_score_matches_thesis_table_2(self) -> None:
        """
        Проверка контрольного примера из Таблицы 2 диссертации:
        Ранги топ-5 направлений: [97, 92, 88, 81, 78] -> Score(C_ENG) = 87.2%
        """
        ranks = [97.0, 92.0, 88.0, 81.0, 78.0, 54.0, 32.0]

        core_score = calculate_core_score(ranks, top_k=5)

        assert math.isclose(core_score, 87.2, abs_tol=1e-2)