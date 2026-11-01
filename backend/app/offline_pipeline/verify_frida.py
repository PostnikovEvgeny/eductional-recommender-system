import logging

from app.offline_pipeline.embedder import FridaEmbedder
from app.offline_pipeline.keybert_extractor import FridaKeyBERTExtractor
from app.offline_pipeline.preprocessor import TextPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SAMPLE_ANNOTATION = """
Технологии интеллектуального анализа данных обеспечивают формирование аналитических данных
посредством выполнения операции очищения данных локальных баз ор- ганизации, применения
статистических методов и других сложных алгоритмов. Появлению аналитических систем
способствовало осознание руководящим звеном предприятий факта, что в базах данных содержится
не только информация, но и знания (скрытые закономерно- сти).
Изучаемые объекты дисциплины: Задача классификации, Задача регрессии, Задача прогнозирования,
Задача кластеризации, Задача анализа последовательностей.
Краткое содержание: Задача анализа отклонений, фильтрация ошибок ввода, поиск мошеннических
операций. Прогнозирование цен акций с использованием числовых последовательностей.
Предсказание значения числового параметра, решение задачи регрессии, решение задачи классификации.
Построение кластеров, деление множества объектов. Автоматическое изучение страниц пользователей
web-сайтов, анализ последовательности переходов, сиквенциальный анализ.
"""


def main() -> None:
    preprocessor = TextPreprocessor()
    embedder = FridaEmbedder()
    extractor = FridaKeyBERTExtractor(preprocessor=preprocessor, embedder=embedder)

    print("=== 1. Извлечение ключевых терминов методом KeyBERT на базе FRIDA (без MMR) ===")
    all_ranked = extractor.extract_keywords_from_text(
        SAMPLE_ANNOTATION,
        top_n=50,
        min_semantic_score=0.0,
        use_mmr=False,
    )

    print("\nТОП-10 САМЫХ ВАЖНЫХ ТЕРМИНОВ (по мнению нейросети FRIDA):")
    for kw in all_ranked[:10]:
        print(f"  [+] {kw.term:<38} | близость к тексту: {kw.semantic_score:.4f} | частота: {kw.frequency}")

    print("\nАНТИ-ТОП (СЛОВЕСНЫЙ ШУМ, который нейросеть автоматически отбросила вниз):")
    for kw in all_ranked[-7:]:
        print(f"  [-] {kw.term:<38} | близость к тексту: {kw.semantic_score:.4f} | частота: {kw.frequency}")

    print("\n=== 2. Финальный отбор ТОП-8 терминов с диверсификацией MMR (KeyBERT + MMR) ===")
    mmr_keywords = extractor.extract_keywords_from_text(
        SAMPLE_ANNOTATION,
        top_n=8,
        min_semantic_score=0.25,
        use_mmr=True,
        diversity=0.35,
    )
    for idx, kw in enumerate(mmr_keywords, start=1):
        print(f"  {idx}. {kw.term:<38} (score: {kw.semantic_score:.4f})")


if __name__ == "__main__":
    main()