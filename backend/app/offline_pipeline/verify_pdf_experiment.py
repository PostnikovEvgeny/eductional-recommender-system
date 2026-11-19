import logging
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from app.core.config import ROOT_DIR
from app.offline_pipeline.clusterer import (
    ClusteredDomain,
    ClusteringConfig,
    HierarchicalDomainClusterer,
)
from app.offline_pipeline.embedder import FridaEmbedder
from app.offline_pipeline.keybert_extractor import ExtractedKeyword, FridaKeyBERTExtractor
from app.offline_pipeline.opop_parser import ParsedOpopSummary, PnrpuOpopParser
from app.offline_pipeline.pdf_parser import ParsedDisciplinePdf, PnrpuAnnotationPdfParser
from app.offline_pipeline.preprocessor import TextPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DirectionExperimentProfile:
    """Агрегированный семантический профиль направления по итогам обработки ОПОП + PDF."""

    folder_name: str
    code: str
    name: str
    macro_title: str | None
    disciplines: list[ParsedDisciplinePdf]
    direction_centroid_doc: np.ndarray
    keywords: list[ExtractedKeyword]
    hierarchy: dict[int, list[ClusteredDomain]]


def _parse_folder_title(folder_name: str) -> tuple[str, str]:
    parts = folder_name.split("_", maxsplit=1)
    if len(parts) == 2 and "." in parts[0]:
        return parts[0], parts[1].replace("_", " ")
    return "00.00.00", folder_name.replace("_", " ")


def _deduplicate_direction_keywords(
    keywords: list[ExtractedKeyword],
) -> list[ExtractedKeyword]:
    """Склеивает близкие по корням термины-клоны в пуле направления."""
    sorted_kws = sorted(
        keywords,
        key=lambda k: (k.is_anchor, k.semantic_score, len(k.term)),
        reverse=True,
    )
    deduped: list[ExtractedKeyword] = []

    for cand in sorted_kws:
        is_clone = False
        for idx, existing in enumerate(deduped):
            overlap = FridaKeyBERTExtractor._lexical_overlap(cand.term, existing.term)
            is_substring = cand.term in existing.term or existing.term in cand.term
            if is_substring or overlap >= 0.75:
                merged_vec = 0.6 * existing.embedding + 0.4 * cand.embedding
                merged_vec = merged_vec / np.linalg.norm(merged_vec)
                deduped[idx] = ExtractedKeyword(
                    term=existing.term,
                    semantic_score=round(
                        min(
                            1.0,
                            max(existing.semantic_score, cand.semantic_score) + 0.02,
                        ),
                        4,
                    ),
                    frequency=existing.frequency + cand.frequency,
                    embedding=np.asarray(merged_vec, dtype=np.float32),
                    source_discipline=existing.source_discipline,
                    is_anchor=(existing.is_anchor or cand.is_anchor),
                )
                is_clone = True
                break

        if not is_clone:
            deduped.append(cand)

    return deduped


def main() -> None:
    t_start = time.perf_counter()

    pdfs_root = ROOT_DIR / "backend" / "data" / "pdfs"
    opop_root = ROOT_DIR / "backend" / "data" / "opop"

    pdf_parser = PnrpuAnnotationPdfParser()
    opop_parser = PnrpuOpopParser()
    preprocessor = TextPreprocessor()
    embedder = FridaEmbedder()
    extractor = FridaKeyBERTExtractor(preprocessor=preprocessor, embedder=embedder)
    clusterer = HierarchicalDomainClusterer()

    # 1. Читаем паспорта ОПОП
    opop_by_code: dict[str, ParsedOpopSummary] = {}
    opop_by_name: dict[str, ParsedOpopSummary] = {}
    if opop_root.is_dir():
        for opop_pdf in sorted(opop_root.glob("*.pdf")):
            summary = opop_parser.parse_opop_pdf(opop_pdf)
            if summary.direction_code:
                opop_by_code[summary.direction_code] = summary
            opop_by_name[opop_pdf.stem.lower()] = summary

    # 2. Парсим аннотации дисциплин
    direction_folders = sorted([p for p in pdfs_root.iterdir() if p.is_dir()])
    direction_docs: dict[str, list[ParsedDisciplinePdf]] = {}
    corpus_doc_frequency: Counter[str] = Counter()
    all_unique_candidates: set[str] = set()
    total_pdfs = 0

    for folder in direction_folders:
        docs = pdf_parser.parse_direction_folder(folder)
        direction_docs[folder.name] = docs
        total_pdfs += len(docs)
        for doc in docs:
            doc_counts = preprocessor.extract_candidate_terms(doc.clean_content_text)
            title_counts = preprocessor.extract_candidate_terms(
                doc.discipline_name, allow_procedural_head=True
            )
            top_doc_terms = {t for t, _ in doc_counts.most_common(25)} | set(
                title_counts.keys()
            )
            all_unique_candidates.update(top_doc_terms)
            for t in doc_counts:
                corpus_doc_frequency[t] += 1

    max_allowed_doc_freq = max(3, int(total_pdfs * 0.35))

    # 3. Глобальный батч-инференс FRIDA (выполняется 1 раз при загрузке PDF)
    all_dense_texts: list[str] = []
    doc_slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    for folder_name, docs in direction_docs.items():
        dense_list = [
            preprocessor.clean_for_dense(d.clean_content_text)[:3000] for d in docs
        ]
        all_dense_texts.extend(dense_list)
        doc_slices[folder_name] = (cursor, cursor + len(dense_list))
        cursor += len(dense_list)

    logger.info(
        "Запуск глобального батч-кодирования: %d дисциплин и %d кандидатов...",
        len(all_dense_texts),
        len(all_unique_candidates),
    )
    all_doc_vecs_search = embedder.encode_documents(all_dense_texts, batch_size=64)
    all_doc_vecs_topic = embedder.encode_texts(
        all_dense_texts, prefix="categorize_topic: ", batch_size=64
    )

    sorted_candidates = sorted(all_unique_candidates)
    all_cand_vecs = embedder.encode_texts(
        sorted_candidates, prefix="categorize_topic: ", batch_size=128
    )
    global_cand_cache: dict[str, np.ndarray] = dict(
        zip(sorted_candidates, all_cand_vecs, strict=True)
    )

    profiles: list[DirectionExperimentProfile] = []

    print("\n" + "=" * 90)
    print("ЭТАП 1. ПРЕДПРОСМОТР КАРТОЧЕК КЛАСТЕРОВ ДЛЯ ПАНЕЛИ ИНЖЕНЕРА-МЕТОДИСТА (HITL)")
    print("=" * 90)

    for folder_name, docs in direction_docs.items():
        if not docs:
            continue

        dir_code, dir_name = _parse_folder_title(folder_name)
        start_i, end_i = doc_slices[folder_name]
        dir_search_vecs = all_doc_vecs_search[start_i:end_i]
        dir_topic_vecs = all_doc_vecs_topic[start_i:end_i]

        direction_centroid_doc = embedder.calculate_direction_centroid(dir_search_vecs)
        opop_summary = opop_by_code.get(dir_code) or opop_by_name.get(dir_name.lower())

        macro_title: str | None = None
        if opop_summary and opop_summary.profiles:
            macro_title = " / ".join(opop_summary.profiles[:2])

        print(f"\n{'*' * 90}")
        print(
            f"НАПРАВЛЕНИЕ: {dir_code} «{dir_name}»\n"
            f"Профили из ОПОП: {opop_summary.profiles if opop_summary else 'Не найдены'}"
        )
        print(f"{'*' * 90}")

        pooled_keywords: dict[str, ExtractedKeyword] = {}

        for doc, disc_topic_vec in zip(docs, dir_topic_vecs, strict=True):
            kws = extractor.extract_keywords_from_text(
                doc.clean_content_text,
                discipline_title=doc.discipline_name,
                excluded_terms={dir_name.lower()},
                precomputed_doc_vec=disc_topic_vec,
                candidate_vec_cache=global_cand_cache,
                top_n=5,
                min_semantic_score=0.35,
                use_mmr=True,
                diversity=0.40,
            )
            filtered_kws = [
                k for k in kws if corpus_doc_frequency[k.term] <= max_allowed_doc_freq
            ] or kws

            for kw in filtered_kws:
                ctx_vec = 0.75 * kw.embedding + 0.25 * disc_topic_vec
                ctx_vec = ctx_vec / np.linalg.norm(ctx_vec)

                if kw.term in pooled_keywords:
                    prev = pooled_keywords[kw.term]
                    boosted_score = min(
                        1.0, max(prev.semantic_score, kw.semantic_score) + 0.04
                    )
                    avg_vec = 0.5 * prev.embedding + 0.5 * ctx_vec
                    avg_vec = avg_vec / np.linalg.norm(avg_vec)
                    pooled_keywords[kw.term] = ExtractedKeyword(
                        term=kw.term,
                        semantic_score=round(boosted_score, 4),
                        frequency=prev.frequency + kw.frequency,
                        embedding=np.asarray(avg_vec, dtype=np.float32),
                        source_discipline=prev.source_discipline,
                        is_anchor=True,
                    )
                else:
                    pooled_keywords[kw.term] = ExtractedKeyword(
                        term=kw.term,
                        semantic_score=kw.semantic_score,
                        frequency=kw.frequency,
                        embedding=np.asarray(ctx_vec, dtype=np.float32),
                        source_discipline=kw.source_discipline,
                        is_anchor=kw.is_anchor,
                    )

        final_keywords = _deduplicate_direction_keywords(
            list(pooled_keywords.values())
        )
        hierarchy = clusterer.build_multiscale_hierarchy(
            final_keywords,
            levels=(5, 3, 1),
            macro_direction_title=macro_title,
        )

        for level in (5, 3, 1):
            print(f"\n  --- УРОВЕНЬ СЖАТИЯ {level} ---")
            for b_idx, dom in enumerate(hierarchy[level], start=1):
                alts_str = " | ".join(dom.alternative_labels) if dom.alternative_labels else "-"
                discs_str = ", ".join(dom.source_disciplines[:3])
                print(
                    f"    Кластер {b_idx}: [{dom.name}] "
                    f"(терминов: {len(dom.member_terms)}, плотность: {dom.cohesion_score:.3f})"
                )
                print(f"       Альтернативы (1 клик): {alts_str}")
                print(f"       Ведущие дисциплины   : {discs_str}")

        profiles.append(
            DirectionExperimentProfile(
                folder_name=folder_name,
                code=dir_code,
                name=dir_name,
                macro_title=macro_title,
                disciplines=docs,
                direction_centroid_doc=direction_centroid_doc,
                keywords=final_keywords,
                hierarchy=hierarchy,
            )
        )

    # 4. Бенчмарк мгновенного пересчёта при движении ползунков инженером (по кэшу)
    print("\n" + "=" * 90)
    print("ЭТАП 2. ТЕСТ МГНОВЕННОГО ПЕРЕСЧЕТА ПРИ ИЗМЕНЕНИИ ПОЛЗУНКОВ (ClusteringConfig)")
    print("=" * 90)

    custom_cfg = ClusteringConfig(
        anchor_weight=1.45,
        rebalance_clusters=False,
        allow_composite_level_3=False,
        broad_term_penalty=0.65,
    )
    t_recluster_start = time.perf_counter()
    for prof in profiles:
        _ = clusterer.build_multiscale_hierarchy(
            prof.keywords,
            levels=(5, 3, 1),
            macro_direction_title=prof.macro_title,
            config=custom_cfg,
        )
    recluster_ms = (time.perf_counter() - t_recluster_start) * 1000
    print(
        f"Пересчёт иерархии 1-3-5 сразу для всех {len(profiles)} направлений "
        f"по кэшу векторов занял: {recluster_ms:.2f} мс ({recluster_ms / 1000:.4f} сек.)"
    )

    total_elapsed = time.perf_counter() - t_start
    print(f"\n[ГОТОВО] Полное время первичного прогона (FRIDA + PDF + ОПОП): {total_elapsed:.2f} сек.")


if __name__ == "__main__":
    main()