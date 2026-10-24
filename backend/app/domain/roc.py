from app.core.exceptions import InvalidDomainsCountError

MIN_ROC_DOMAINS = 3
MAX_ROC_DOMAINS = 5


def calculate_roc_weights(
    k: int,
    *,
    enforce_bounds: bool = True,
    min_k: int = MIN_ROC_DOMAINS,
    max_k: int = MAX_ROC_DOMAINS,
) -> list[float]:
    """
    Рассчитывает весовые коэффициенты важности критериев методом ROC (Rank Order Centroid).

    Формула: W_i = (1 / k) * sum_{j=i..k} (1 / j), где i in [1..k] — ранг интереса.
    Сумма всех весов всегда строго равна 1.0.

    :param k: Количество ранжированных интересов, выбранных абитуриентом.
    :param enforce_bounds: Проверять ли строгое ограничение диапазона 3 <= k <= 5.
    :param min_k: Минимально допустимое количество интересов (по умолчанию 3).
    :param max_k: Максимально допустимое количество интересов (по умолчанию 5).
    :return: Упорядоченный по убыванию список весов [W_1, W_2, ..., W_k].
    """
    if k <= 0:
        raise InvalidDomainsCountError(count=k, min_allowed=min_k, max_allowed=max_k)

    if enforce_bounds and not (min_k <= k <= max_k):
        raise InvalidDomainsCountError(count=k, min_allowed=min_k, max_allowed=max_k)

    weights: list[float] = []
    for i in range(1, k + 1):
        harmonic_tail = sum(1.0 / j for j in range(i, k + 1))
        weight = harmonic_tail / k
        weights.append(weight)

    return weights


def map_domains_to_roc_weights(
    ordered_domain_ids: list[int],
    *,
    enforce_bounds: bool = True,
) -> dict[int, float]:
    """
    Сопоставляет упорядоченный по приоритету список ID доменов с их весами ROC.

    :param ordered_domain_ids: Список ID доменов от наиболее важного (ранг 1) к наименее важному.
    :param enforce_bounds: Проверять ли ограничение 3 <= k <= 5.
    :return: Словарь {domain_id: weight}.
    """
    # Проверка на отсутствие дубликатов ID доменов в выборе пользователя
    if len(set(ordered_domain_ids)) != len(ordered_domain_ids):
        raise ValueError("Список выбранных областей интереса не должен содержать дубликатов.")

    weights = calculate_roc_weights(
        len(ordered_domain_ids), enforce_bounds=enforce_bounds
    )
    return dict(zip(ordered_domain_ids, weights, strict=True))