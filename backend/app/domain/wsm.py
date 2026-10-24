from collections.abc import Sequence

from app.core.exceptions import DimensionMismatchError, InvalidAffiliationValueError

EPSILON = 1e-6


def calculate_wsm_score(
    weights: Sequence[float],
    affiliations: Sequence[float],
) -> float:
    """
    Вычисляет интегральную оценку направления подготовки методом взвешенной суммы (WSM).

    Формула: Score_D = sum_{i=1..k} (W_i * mu_D(domain_i))

    :param weights: Весовые коэффициенты важности интересов (сумма равна 1.0).
    :param affiliations: Значения функции принадлежности mu_D in [0, 1] для каждого интереса.
    :return: Итоговая оценка соответствия направления Score_D in [0.0, 1.0].
    """
    if len(weights) != len(affiliations):
        raise DimensionMismatchError(
            weights_len=len(weights), affiliations_len=len(affiliations)
        )

    if not weights:
        return 0.0

    total_score = 0.0
    for w, mu in zip(weights, affiliations, strict=True):
        # Допускаем микропогрешность float на границах 0.0 и 1.0
        if mu < -EPSILON or mu > 1.0 + EPSILON:
            raise InvalidAffiliationValueError(value=mu)

        clamped_mu = max(0.0, min(1.0, mu))
        total_score += w * clamped_mu

    return float(total_score)


def calculate_wsm_with_contributions(
    domain_weights: dict[int, float],
    domain_affiliations: dict[int, float],
) -> tuple[float, dict[int, float]]:
    """
    Рассчитывает итоговую оценку WSM и детализированный вклад каждого интереса
    для обеспечения объяснимости (интерпретируемости) рекомендации в UI.

    :param domain_weights: Словарь {domain_id: W_i}.
    :param domain_affiliations: Словарь {domain_id: mu_D(domain_i)}.
    :return: Кортеж (итоговый Score_D, словарь вкладов {domain_id: W_i * mu_i}).
    """
    weights_list: list[float] = []
    affiliations_list: list[float] = []
    contributions: dict[int, float] = {}

    for domain_id, weight in domain_weights.items():
        # Если для домена нет записи близости, считаем принадлежность равной 0.0
        mu = domain_affiliations.get(domain_id, 0.0)
        if mu < -EPSILON or mu > 1.0 + EPSILON:
            raise InvalidAffiliationValueError(value=mu)

        clamped_mu = max(0.0, min(1.0, mu))
        weights_list.append(weight)
        affiliations_list.append(clamped_mu)
        contributions[domain_id] = weight * clamped_mu

    total_score = calculate_wsm_score(weights_list, affiliations_list)
    return total_score, contributions