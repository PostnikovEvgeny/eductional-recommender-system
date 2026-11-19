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
    Извлекает только содержательную предметную часть ('Цели и задачи',
    'Изучаемые объекты', 'Краткое содержание разделов'), полностью удаляя
    таблицы часов, шапки страниц и коды компетенций.
    """

    # Регулярные выражения для удаления табличных колонок и канцелярских штампов шаблона РПД
    _NOISE_PATTERNS: tuple[str, ...] = (
        r"(?is)аннотация дисциплины\s*[«\"].*?по направлению\s*[«\"].*?[»\"]\.?",
        r"(?is)(объем и виды учебной работы|вид учебной работы\s*всего часов).*?(краткое содержание дисциплины|наименование разделов дисциплины)",
        r"(?i)вид учебной работы\s*всего часов\s*распределение по семестрам в часах\s*номер семестра",
        r"(?i)наименование разделов дисциплины\s*с кратким\s*содержанием",
        r"(?i)объем аудиторных\s*занятий по видам\s*(в\s*часах)?",
        r"(?i)объем внеаудиторных\s*занятий по видам\s*(в\s*часах)?",
        r"(?i)итого по\s*\d*\s*-?\s*му семестру",
        r"(?i)итого по дисциплине",
        r"(?i)краткое содержание дисциплины",
        r"(?i)изучаемые объекты дисциплины",
        r"(?i)цели и задачи( учебной)? дисциплины",
        r"(?i)цель( учебной| изучения| освоения)? дисциплины\s*[-–:]?",
        r"(?i)задачи( учебной| изучения)? дисциплины\s*[-–:]?",
        r"(?i)предметом (освоения|изучения) дисциплины являются следующие объекты\s*[-–:]?",
        r"(?i)в процессе изучения данной дисциплины студент расширяет и углубляет следующие компетенции\s*[-–:]?",
        r"(?i)\b(УК|ОПК|ПК)\s*-\s*\d*[\.,:\s]*",
        r"(?i)\b\d*\s*-?\s*й семестр\b",
        r"(?i)\b(Модуль|Раздел|Тема)\s*\d*[\.,:\s]+",
        r"\b(Л|ЛР|ПЗ|СРС|CPC|КСР|КП|КР)\b",
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
        """Извлекает название дисциплины, шифр и название направления из шапки 1-й страницы."""
        header_slice = full_text[:900]

        disc_match = re.search(
            r"(?i)аннотация дисциплины\s*[«\"]([^»\"\n]+)[»\"]", header_slice
        )
        if not disc_match:
            disc_match = re.search(
                r"(?i)дисциплина\s*[«\"]([^»\"\n]+)[»\"]", header_slice
            )

        raw_disc_name = (
            disc_match.group(1).strip() if disc_match else fallback_filename
        )
        discipline_name = re.sub(r"\s+", " ", raw_disc_name)
        discipline_name = re.sub(r"(?i)усстройств", "устройств", discipline_name)

        dir_match = re.search(
            r"(?i)по\s+направлению\s*[«\"](\d{2}\.\d{2}\.\d{2})\s+([^»\"\n]+)[»\"]",
            header_slice,
        )
        direction_code = dir_match.group(1).strip() if dir_match else None
        direction_name = (
            re.sub(r"\s+", " ", dir_match.group(2).strip()) if dir_match else None
        )

        return discipline_name, direction_code, direction_name

    def _clean_semantic_sections(self, full_text: str) -> str:
        """
        Оставляет только содержательные предметные разделы и очищает их от таблиц,
        переносов строк, цифр часов и служебных заголовков.
        """
        text = full_text

        # Исправляем частые опечатки исходных документов ПНИПУ
        text = re.sub(r"(?i)усстройств", "устройств", text)
        text = re.sub(r"(?i)инеграци", "интеграци", text)

        # 1. Склеиваем слова, разорванные переносом строки в узких ячейках таблиц PDF
        text = re.sub(r"([а-яёА-ЯЁ]{2,})-\s*\n\s*([а-яёА-ЯЁ]{2,})", r"\1\2", text)
        text = re.sub(r"([а-яёА-ЯЁ]{2,})-\s+([а-яёА-ЯЁ]{2,})", r"\1\2", text)

        # 2. Отрезаем вводную шапку документа до раздела "Цели и задачи"
        goals_match = re.search(
            r"(?i)(цели и задачи|цель\s+(учебной|изучения|освоения)\s+дисциплины|цель\s*[–-])",
            text,
        )
        if goals_match:
            text = text[goals_match.start() :]

        # 3. Применяем все паттерны очистки таблиц и шаблонных заголовков
        for pattern in self._NOISE_PATTERNS:
            text = re.sub(pattern, ". ", text)

        # 4. Удаляем отдельно стоящие числа (часы лекций/лабораторных)
        text = re.sub(r"(?<!\w)\d+(?!\w)", " ", text)

        # 5. Заменяем переносы строк и маркеры списков на пробелы/точки
        text = text.replace("\n", " ")
        text = re.sub(r"[•●▪■◦]", ". ", text)

        # 6. Нормализуем пробелы и пунктуацию
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[\.,;:\-\s]{2,}", ". ", text)
        text = text.lstrip(".,;:- ")

        return text

    def parse_pdf(self, pdf_path: str | Path) -> ParsedDisciplinePdf:
        """Полный цикл разбора одного PDF-файла аннотации дисциплины."""
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
        """Парсит все PDF-файлы аннотаций дисциплин внутри папки конкретного направления."""
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