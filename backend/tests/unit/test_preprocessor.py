from app.offline_pipeline.preprocessor import TextPreprocessor

SAMPLE_ITAD_ANNOTATION = """
Технологии интеллектуального анализа данных обеспечивают формирование аналитических данных
посредством выполнения операции очищения данных локальных баз ор- ганизации, применения
статистических методов и других сложных алгоритмов.
Изучаемые объекты дисциплины: Задача классификации, Задача регрессии, Задача кластеризации,
прогнозирование цен акций, сиквенциальный анализ, поиск мошеннических операций.
"""


class TestTextPreprocessor:
    """Тестирование двухпоточного лингвистического препроцессора."""

    def test_clean_for_dense_fixes_hyphenation_and_html(self) -> None:
        """Проверка склейки разорванных переносами слов из PDF и очистки HTML."""
        preprocessor = TextPreprocessor()
        raw = "<p>Очищение локальных баз ор- ганизации и закономерно- сти.</p>"

        cleaned = preprocessor.clean_for_dense(raw)

        assert "<p>" not in cleaned
        assert "организации" in cleaned
        assert "закономерности" in cleaned

    def test_extract_candidate_terms_from_real_itad_annotation(self) -> None:
        """Проверка извлечения осмысленных терминов из реальной аннотации ИТАД."""
        preprocessor = TextPreprocessor()

        candidates = preprocessor.extract_candidate_terms(SAMPLE_ITAD_ANNOTATION)

        # Проверяем, что выделены ключевые предметные понятия в правильной форме
        assert "интеллектуальный анализ данных" in candidates
        assert "сиквенциальный анализ" in candidates
        assert "мошеннический операция" in candidates or "классификация" in candidates
        assert "кластеризация" in candidates
        assert "регрессия" in candidates

        # Проверяем, что академические стоп-слова отфильтрованы
        assert "дисциплина" not in candidates
        assert "задача" not in candidates
        assert "применение" not in candidates
        assert "формирование" not in candidates