import logging
import numpy as np

from app.offline_pipeline.clusterer import HierarchicalDomainClusterer
from app.offline_pipeline.embedder import FridaEmbedder
from app.offline_pipeline.keybert_extractor import ExtractedKeyword, FridaKeyBERTExtractor
from app.offline_pipeline.preprocessor import TextPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SAMPLE_DIRECTION_DISCIPLINES = [
    """
    Интеллектуальные технологии анализа данных. Машинное обучение, нейронные сети и глубокое обучение.
    Решение задачи классификации, задачи регрессии, задачи кластеризации. Прогнозирование временных рядов,
    анализ последовательностей и поиск скрытых закономерностей в больших данных.
    """,
    """
    Технологии баз данных и СУБД. Проектирование реляционных баз данных, нормализация отношений,
    язык запросов SQL, транзакции и оптимизация индексов. Распределенные хранилища данных и NoSQL системы.
    """,
    """
    Веб-программирование и интернет-технологии. Разработка клиент-серверных приложений, архитектура REST API,
    микросервисная архитектура, фронтенд и бэкенд разработка веб-приложений, протокол HTTP.
    """,
    """
    Разработка мобильных приложений. Программирование под мобильные платформы Android и iOS,
    проектирование пользовательского интерфейса мобильных устройств, кроссплатформенная мобильная разработка.
    """,
    """
    Информационная безопасность и защита информации. Криптографические методы защиты данных,
    шифрование трафика, сетевая безопасность, аутентификация пользователей и анализ уязвимостей программного обеспечения.
    """,
]


def main() -> None:
    preprocessor = TextPreprocessor()
    embedder = FridaEmbedder()
    extractor = FridaKeyBERTExtractor(preprocessor=preprocessor, embedder=embedder)
    clusterer = HierarchicalDomainClusterer()

    print("=== 1. Извлечение контекстуализированных ключевых терминов из 5 дисциплин ===")
    pooled_keywords: dict[str, ExtractedKeyword] = {}

    for idx, disc_text in enumerate(SAMPLE_DIRECTION_DISCIPLINES, start=1):
        dense_disc = preprocessor.clean_for_dense(disc_text)
        disc_vec = embedder.encode_texts([dense_disc], prefix="categorize_topic: ")[0]

        kws = extractor.extract_keywords_from_text(
            disc_text, top_n=5, min_semantic_score=0.35, use_mmr=True, diversity=0.35
        )
        print(f"  Дисциплина {idx}: {[k.term for k in kws]}")
        for kw in kws:
            # Обогащаем вектор термина контекстом его дисциплины (70% термин + 30% дисциплина)
            contextual_vec = 0.70 * kw.embedding + 0.30 * disc_vec
            contextual_vec = contextual_vec / np.linalg.norm(contextual_vec)

            contextual_kw = ExtractedKeyword(
                term=kw.term,
                semantic_score=kw.semantic_score,
                frequency=kw.frequency,
                embedding=np.asarray(contextual_vec, dtype=np.float32),
            )
            if kw.term not in pooled_keywords or kw.semantic_score > pooled_keywords[kw.term].semantic_score:
                pooled_keywords[kw.term] = contextual_kw

    final_keywords = list(pooled_keywords.values())
    print(f"\nВсего уникальных терминов-кандидатов направления: {len(final_keywords)}")

    print("\n=== 2. Иерархическое семантическое сжатие HAC (Уровни 5 -> 3 -> 1) ===")
    hierarchy = clusterer.build_multiscale_hierarchy(
        final_keywords,
        levels=(5, 3, 1),
    )

    for level in (5, 3, 1):
        print(f"\n--- УРОВЕНЬ СЖАТИЯ: {level} (кнопок для абитуриента: {len(hierarchy[level])}) ---")
        for i, dom in enumerate(hierarchy[level], start=1):
            print(f"  Кнопка {i}: [{dom.name}] (плотность: {dom.cohesion_score:.3f})")
            print(f"     Входящие термины ({len(dom.member_terms)}): {', '.join(dom.member_terms)}")


if __name__ == "__main__":
    main()