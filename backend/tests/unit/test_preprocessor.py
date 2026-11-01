from app.offline_pipeline.keybert_extractor import FridaKeyBERTExtractor
from app.offline_pipeline.preprocessor import TextPreprocessor

SAMPLE_ITAD_ANNOTATION = """
Технологии интеллектуального анализа данных обеспечивают формирование аналитических данных
посредством выполнения операции очищения данных локальных баз ор- ганизации, применения
статистических методов и других сложных алгоритмов.
Изучаемые объекты дисциплины: Задача классификации, Задача регрессии, Задача кластеризации,
прогнозирование цен акций, сиквенциальный анализ, поиск мошеннических операций.
"""


class TestTextPreprocessorAndKeyBERT:
    """Тестирование двухпоточного лингвистического препроцессора и лексического фильтра MMR."""

    def test_clean_for_dense_fixes_hyphenation_and_headers(self) -> None:
        """Проверка склейки разорванных переносами слов из PDF, очистки HTML и заголовков."""
        preprocessor = TextPreprocessor()
        raw = "<p>Изучаемые объекты дисциплины: Очищение баз ор- ганизации и закономерно- сти.</p>"

        cleaned = preprocessor.clean_for_dense(raw)

        assert "<p>" not in cleaned
        assert "Изучаемые объекты дисциплины:" not in cleaned
        assert "организации" in cleaned
        assert "закономерности" in cleaned

    def test_extract_candidate_terms_from_real_itad_annotation(self) -> None:
        """Проверка извлечения осмысленных терминов из реальной аннотации ИТАД."""
        preprocessor = TextPreprocessor()

        candidates = preprocessor.extract_candidate_terms(SAMPLE_ITAD_ANNOTATION)

        assert "интеллектуальный анализ данных" in candidates
        assert "сиквенциальный анализ" in candidates
        assert "классификация" in candidates
        assert "кластеризация" in candidates
        assert "регрессия" in candidates

        # Проверяем, что структурные стоп-слова отфильтрованы
        assert "дисциплина" not in candidates
        assert "задача" not in candidates
        assert "применение" not in candidates

    def test_lexical_overlap_detects_same_root_phrases(self) -> None:
        """Проверка детектора однокоренных слов для алгоритма MMR."""
        overlap_high = FridaKeyBERTExtractor._lexical_overlap(
            "анализ последовательностей", "анализ последовательности"
        )
        overlap_partial = FridaKeyBERTExtractor._lexical_overlap(
            "интеллектуальный анализ данных", "аналитические данные"
        )
        overlap_zero = FridaKeyBERTExtractor._lexical_overlap(
            "база данных", "нейронная сеть"
        )

        assert overlap_high == 1.0
        assert overlap_partial >= 0.5
        assert overlap_zero == 0.0