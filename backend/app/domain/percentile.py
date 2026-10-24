import math
from collections.abc import Mapping

DEFAULT_TOP_K_DIRECTIONS = 5
FLOAT_TOLERANCE = 1e-9


def calculate_percentile_ranks(scores: Mapping[int, float]) -> dict[int, float]:
    """
    Преобразует сырые оценки направлений Score_D в процентильные ранги PR(D_i).

    Формула: PR(D_i) = ((L_i + 0.5 * E_i) / N) * 100,
    где:
      L_i — количество направлений со значением оценки строго меньше Score(D_i),
      E_i — количество направлений с таким же значением оценки (включая само направление),
      N   — общее количество направлений в выборке.

    :param scores: Словарь {direction_id: raw_score}.
    :return: Словарь {direction_id: percentile_rank} в диапазоне [0.0, 100.0].
    """
    n = len(scores)
    if n == 0:
        return {}

    all_values = list(scores.values())
    percentile_ranks: dict[int, float] = {}

    for direction_id, target_score in scores.items():
        less_count = 0
        equal_count = 0

        for val in all_values:
            if math.isclose(val, target_score, abs_tol=FLOAT_TOLERANCE):
                equal_count += 1
            elif val < target_score:
                less_count += 1

        pr = ((less_count + 0.5 * equal_count) / n) * 100.0
        percentile_ranks[direction_id] = round(pr, 2)

    return percentile_ranks


def calculate_core_score(
    direction_percentile_ranks: list[float],
    top_k: int = DEFAULT_TOP_K_DIRECTIONS,
) -> float:
    """
    Вычисляет агрегированную оценку образовательного ядра как среднее арифметическое
    процентильных рангов Top-K лучших направлений, входящих в данное ядро.

    Формула: Score(C_k) = (1 / K) * sum_{D_i in TopK(C_k)} PR(D_i)

    :param direction_percentile_ranks: Список процентильных рангов направлений ядра.
    :param top_k: Количество лучших направлений для усреднения (по умолчанию 5).
    :return: Итоговая оценка образовательного ядра в диапазоне [0.0, 100.0].
    """
    if not direction_percentile_ranks or top_k <= 0:
        return 0.0

    # Сортируем ранги по убыванию и берем срез Top-K
    sorted_ranks = sorted(direction_percentile_ranks, reverse=True)
    top_slice = sorted_ranks[:top_k]

    core_score = sum(top_slice) / len(top_slice)
    return round(core_score, 2)