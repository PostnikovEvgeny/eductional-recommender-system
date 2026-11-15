from app.offline_pipeline.pdf_parser import PnrpuAnnotationPdfParser

SAMPLE_RAW_PDF_TEXT = """
АННОТАЦИЯ ДИСЦИПЛИНЫ
«Базы данных»
Дисциплина «Базы данных» является частью программы бакалавриата
по направлению «09.03.04 Программная инженерия».
Цели и задачи дисциплины
Цель учебной дисциплины освоение принципов построения баз данных.
Изучаемые объекты дисциплины
Предметом освоения дисциплины являются следующие объекты: базы данных и их структуры; модели баз данных.
Объем и виды учебной работы
Вид учебной работы | Всего часов | 324
лекции (Л) | 40 | лабораторные работы (ЛР) | 116
Самостоятельная работа студентов (СРС) | 128
Экзамен | 36 | Курсовая работа (КР) | 18
Краткое содержание дисциплины
Наименование разделов дисциплины с кратким содержанием | Объем аудиторных занятий по видам в часах
3-й семестр
Язык SQL | 16 | 58 | 0 | 96
Селекция и проекция в SQL. Нормализация БД. Реляционная алгебра.
ИТОГО по 3-му семестру | 58 | 20 | 0 | 100
ИТОГО по дисциплине | 116 | 40 | 0 | 128
"""


class TestPnrpuAnnotationPdfParser:
    """Тестирование парсера и очистителя PDF-аннотаций ПНИПУ."""

    def test_extract_metadata_from_header(self) -> None:
        """Проверка извлечения названия дисциплины и шифра направления из шапки."""
        parser = PnrpuAnnotationPdfParser()
        disc_name, dir_code, dir_name = parser._extract_metadata_from_header(
            SAMPLE_RAW_PDF_TEXT, fallback_filename="fallback_name"
        )

        assert disc_name == "Базы данных"
        assert dir_code == "09.03.04"
        assert dir_name == "Программная инженерия"

    def test_clean_semantic_sections_removes_hours_table_and_keeps_topics(self) -> None:
        """Проверка удаления таблицы часов и сохранения тем краткого содержания."""
        parser = PnrpuAnnotationPdfParser()
        cleaned = parser._clean_semantic_sections(SAMPLE_RAW_PDF_TEXT)

        # Проверяем, что содержательные темы остались
        assert "построения баз данных" in cleaned
        assert "Язык SQL" in cleaned
        assert "Нормализация БД" in cleaned
        assert "Реляционная алгебра" in cleaned

        # Проверяем, что табличный шум и часы полностью вырезаны
        assert "Самостоятельная работа студентов" not in cleaned
        assert "Курсовая работа" not in cleaned
        assert "ИТОГО по дисциплине" not in cleaned
        assert "324" not in cleaned