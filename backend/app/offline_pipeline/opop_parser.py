import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedOpopSummary:
    """Извлеченное смысловое ядро из общей характеристики ОПОП."""

    file_path: Path
    direction_code: str | None
    direction_name: str | None
    profiles: list[str]


class PnrpuOpopParser:
    """
    Универсальный парсер общей характеристики ОПОП (Компетентностной модели выпускника ПНИПУ).
    Обходит титульные страницы и предисловия всех профилей внутри PDF-документа,
    игнорируя многостраничные приложения с аудиториями и списком преподавателей.
    """

    @staticmethod
    def _fix_broken_words(text: str) -> str:
        """Убирает случайные пробелы и переносы внутри слов из PDF."""
        fixes = {
            r"мене\s+джмент": "менеджмент",
            r"аэродро\s+мы": "аэродромы",
            r"программно-\s*информационных": "программно-информационных",
            r"архитектурно-\s*строительное": "архитектурно-строительное",
            r"\bИ\b": "и",
        }
        result = text.replace("\n", " ")
        for pattern, replacement in fixes.items():
            result = re.sub(pattern, replacement, result)
        return re.sub(r"\s+", " ", result).strip()

    def parse_opop_pdf(self, pdf_path: str | Path) -> ParsedOpopSummary:
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"Файл ОПОП не найден: {path}")

        reader = PdfReader(str(path))
        total_pages = len(reader.pages)

        title_pages_text: list[str] = []
        pages_scanned = 0

        idx = 0
        while idx < total_pages:
            raw_page = reader.pages[idx].extract_text() or ""
            head_slice = raw_page[:900]
            is_header_page = (
                idx < 4
                or "Предисловие" in head_slice
                or "Направленность (профиль)" in head_slice
                or "2. Основные характеристики образовательной программы" in head_slice
            )
            if is_header_page and raw_page.strip():
                title_pages_text.append(raw_page)
                pages_scanned += 1
                # После прочтения "Предисловия" перепрыгиваем через таблицы компетенций и аудиторий
                if "Предисловие" in head_slice and idx > 0:
                    idx += 35
                    continue
            idx += 1

        combined = "\n".join(title_pages_text)

        # 1. Универсальный поиск шифра направления (XX.XX.XX) без хардкода конкретных специальностей
        code_match = re.search(
            r"(?is)(?:направлени[ею]\s+подготовки|направлени[ею])?[^0-9]{0,25}\b(\d{2}\.\d{2}\.\d{2})\b",
            combined,
        )
        dir_code = code_match.group(1).strip() if code_match else None

        dir_name = path.stem
        if dir_code:
            name_match = re.search(
                rf"{re.escape(dir_code)}\s*[«\"]?([А-ЯЁ][а-яё]+(?:\s+[а-яё]+){{0,3}})",
                combined,
            )
            if name_match:
                dir_name = self._fix_broken_words(name_match.group(1))

        # 2. Извлекаем направленности (профили) программы из Предисловия и Титульного листа
        profiles: list[str] = []
        profile_patterns = (
            r"(?is)программа\s+бакалавриата\s*[«\"]([^»\"]{6,130})[»\"]",
            r"(?is)направленности\s*(?:\(профиля\))?\s*[«\"]([^»\"]{6,130})[»\"]",
        )
        for pat in profile_patterns:
            for prof_match in re.finditer(pat, combined):
                clean_prof = self._fix_broken_words(prof_match.group(1))
                if (
                    clean_prof
                    and "СУОС" not in clean_prof
                    and "высшего образования" not in clean_prof.lower()
                ):
                    short_prof = clean_prof.split(".")[0].strip()
                    if (
                        short_prof
                        and short_prof[0].isupper()
                        and short_prof not in profiles
                    ):
                        profiles.append(short_prof)

        logger.info(
            "Разобран паспорт ОПОП '%s' (код=%s, обработано титульных стр.: %d из %d): профили=%s",
            path.name,
            dir_code,
            pages_scanned,
            total_pages,
            profiles,
        )

        return ParsedOpopSummary(
            file_path=path,
            direction_code=dir_code,
            direction_name=dir_name,
            profiles=profiles,
        )