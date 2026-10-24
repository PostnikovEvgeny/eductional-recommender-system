class DomainError(Exception):
    """Базовое исключение для всех ошибок бизнес-логики и математической модели."""

    def __init__(self, message: str, code: str = "DOMAIN_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(self.message)


class InvalidDomainsCountError(DomainError):
    """Ошибка при выборе количества интересов вне допустимого диапазона ROC."""

    def __init__(self, count: int, min_allowed: int = 3, max_allowed: int = 5) -> None:
        message = (
            f"Недопустимое количество выбранных областей интереса: {count}. "
            f"По условиям модели ROC требуется выбрать от {min_allowed} до {max_allowed} интересов."
        )
        super().__init__(message=message, code="INVALID_DOMAINS_COUNT")


class DimensionMismatchError(DomainError):
    """Ошибка при несовпадении размерностей векторов весов и степеней принадлежности."""

    def __init__(self, weights_len: int, affiliations_len: int) -> None:
        message = (
            f"Размерность вектора весов ({weights_len}) не совпадает "
            f"с размерностью вектора степеней принадлежности ({affiliations_len})."
        )
        super().__init__(message=message, code="DIMENSION_MISMATCH")


class InvalidAffiliationValueError(DomainError):
    """Ошибка при выходе значения функции принадлежности за пределы [0, 1]."""

    def __init__(self, value: float) -> None:
        message = (
            f"Значение функции принадлежности ({value}) выходит за допустимый диапазон [0.0, 1.0]."
        )
        super().__init__(message=message, code="INVALID_AFFILIATION_VALUE")