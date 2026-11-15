import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedDisciplinePdf:
    """Результат парсинга PDF-файла аннотации дисциплины ПНИПУ."""

    file_path: Path
    discipline_name: str
    direction_code: str | None
    direction_name: str | None
    clean_content_text: str


class PnrpuAnnotationPdfParser:
    """
    Парсер типовых PDF-аннотаций рабочих программ дисциплин ПНИПУ.
    Извлекает только содержательную текстовую часть ('Цели и задачи',
    'Изучаемые объекты', 'Краткое содержание'), удаляя таблицы часов и шапки.
    """

    # Шаблоны служебных строк таблиц, не несущих предметной семантики
    _TABLE_NOISE_PATTERNS: tuple[str, ...] = (
        r"(?i)наименование разделов дисциплины\s*с кратким\s*содержанием",
        r"(?i)объем аудиторных\s*занятий по видам\s*(в\s*часах)?",
        r"(?i)объем внеаудиторных\s*занятий по видам\s*(в\s*часах)?",
        r"(?i)итого по\s*\d+\s*-?\s*му семестру",
        r"(?i)итого по дисциплине",
        r"(?i)\b\d+\s*-?\s*й семестр\b",
        r"\b(Л|ЛР|ПЗ|СРС|CPC|КСР)\b",
    )

    def extract_raw_text_from_pdf(self, pdf_path: str | Path) -> str:
        """Считывает текстовый слой со всех страниц PDF-документа."""
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF-файл не найден: {path}")

        reader = PdfReader(str(path))
        pages_text: list[str] = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                pages_text.append(extracted)

        return "\n".join(pages_text)

    @staticmethod
    def _extract_metadata_from_header(
        full_text: str, fallback_filename: str
    ) -> tuple[str, str | None, str | None]:
        """
        Извлекает название дисциплины, шифр и название направления из шапки 1-й страницы.
        Пример из PDF:
          АННОТАЦИЯ ДИСЦИПЛИНЫ «Базы данных»
          ...по направлению «09.03.04 Программная инженерия».
        """
        header_slice = full_text[:800]

        # 1. Ищем название дисциплины после слов "АННОТАЦИЯ ДИСЦИПЛИНЫ" или "Дисциплина"
        disc_match = re.search(
            r"(?i)аннотация дисциплины\s*[«\"]([^»\"\n]+)[»\"]", header_slice
        )
        if not disc_match:
            disc_match = re.search(
                r"(?i)дисциплина\s*[«\"]([^»\"\n]+)[»\"]", header_slice
            )

        discipline_name = (
            disc_match.group(1).strip() if disc_match else fallback_filename
        )

        # 2. Ищем шифр и название направления подготовки
        dir_match = re.search(
            r"(?i)по\s+направлению\s*[«\"](\d{2}\.\d{2}\.\d{2})\s+([^»\"\n]+)[»\"]",
            header_slice,
        )
        direction_code = dir_match.group(1).strip() if dir_match else None
        direction_name = dir_match.group(2).strip() if dir_match else None

        return discipline_name, direction_code, direction_name

    def _clean_semantic_sections(self, full_text: str) -> str:
        """
        Оставляет только содержательные разделы ('Цели и задачи', 'Изучаемые объекты',
        'Краткое содержание дисциплины') и очищает их от табличных колонок и цифр часов.
        """
        text = full_text

        # 1. Отрезаем вводную шапку документа до раздела "Цели и задачи дисциплины"
        goals_match = re.search(r"(?i)цели и задачи дисциплины", text)
        if goals_match:
            text = text[goals_match.end() :]

        # 2. Вырезаем целиком таблицу часов "Объем и виды учебной работы" на 1-й странице
        # (всё от "Объем и виды учебной работы" до "Краткое содержание дисциплины")
        text = re.sub(
            r"(?is)объем и виды учебной работы.*?краткое содержание дисциплины",
            " ",
            text,
        )

        # 3. Удаляем повторяющиеся шапки таблиц на 2-й и 3-й страницах
        for pattern in self._TABLE_NOISE_PATTERNS:
            text = re.sub(pattern, " ", text)

        # 4. Удаляем отдельно стоящие числа (часы лекций/лабораторных в ячейках таблиц)
        text = re.sub(r"(?<!\w)\d+(?!\w)", " ", text)

        # 5. Склеиваем слова, разорванные дефисом при переносе строк в узких колонках PDF
        text = re.sub(r"([а-яёА-ЯЁ]{2,})-\s*\n\s*([а-яёА-ЯЁ]{2,})", r"\1\2", text)
        text = re.sub(r"([а-яёА-ЯЁ]{2,})-\s+([а-яёА-ЯЁ]{2,})", r"\1\2", text)

        # 6. Убираем остаточные заголовки разделов и лишние пробелы
        text = re.sub(
            r"(?i)(цель учебной дисциплины|задачи учебной дисциплины|изучаемые объекты дисциплины|"
            r"предметом освоения дисциплины являются следующие объекты|краткое содержание дисциплины)\s*:?",
            ". ",
            text,
        )
        text = re.sub(r"\s+", " ", text).strip()
        # Убираем множественные точки подряд (".." -> ".")
        text = re.sub(r"\.{2,}", ".", text)

        return text

    def parse_pdf(self, pdf_path: str | Path) -> ParsedDisciplinePdf:
        """
        Полный цикл разбора одного PDF-файла аннотации дисциплины.
        """
        path = Path(pdf_path)
        raw_text = self.extract_raw_text_from_pdf(path)

        discipline_name, direction_code, direction_name = (
            self._extract_metadata_from_header(raw_text, fallback_filename=path.stem)
        )
        clean_text = self._clean_semantic_sections(raw_text)

        logger.info(
            "Успешно разобран PDF '%s' -> Дисциплина: '%s' (символов текста: %d)",
            path.name,
            discipline_name,
            len(clean_text),
        )

        return ParsedDisciplinePdf(
            file_path=path,
            discipline_name=discipline_name,
            direction_code=direction_code,
            direction_name=direction_name,
            clean_content_text=clean_text,
        )

    def parse_direction_folder(
        self, folder_path: str | Path
    ) -> list[ParsedDisciplinePdf]:
        """
        Парсит все PDF-файлы аннотаций дисциплин внутри папки конкретного направления.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise NotADirectoryError(f"Директория с PDF не найдена: {folder}")

        pdf_files = sorted(folder.glob("*.pdf"))
        results: list[ParsedDisciplinePdf] = []

        for pdf_file in pdf_files:
            try:
                parsed = self.parse_pdf(pdf_file)
                if parsed.clean_content_text:
                    results.append(parsed)
            except Exception as exc:
                logger.error("Ошибка при чтении PDF %s: %s", pdf_file.name, exc)

        return results