import re
from collections import Counter
from dataclasses import dataclass

import pymorphy3

# Общеязыковые стоп-слова и служебные частицы (хранятся через 'е', проверка нормализует 'ё' -> 'е')
GENERAL_STOPWORDS: set[str] = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
    "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над",
    "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая",
    "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
    "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более",
    "всегда", "конечно", "всю", "между", "который", "данная", "данный", "также",
    "свой", "помощь", "посредством", "путем", "включая", "рамка", "случай",
}

# Универсальные академические и канцелярские слова учебных планов (РПД/ОПОП),
# общие для всех факультетов вуза без привязки к конкретной специальности
ACADEMIC_STOPWORDS: set[str] = {
    "студент", "обучающийся", "выпускник", "дисциплина", "семестр", "курс", "лекция",
    "практика", "лабораторный", "работа", "занятие", "зачет", "экзамен", "курсовой",
    "аттестация", "компетенция", "индикатор", "знать", "уметь", "владеть",
    "навык", "цель", "объект", "предмет", "раздел", "тема", "модуль",
    "изучение", "освоение", "формирование", "применение", "использование",
    "выполнение", "решение", "основа", "введение", "понятие", "область", "сфера",
    "способность", "готовность", "деятельность", "профессиональный", "учебный",
    "образовательный", "программа", "направление", "направленность", "подготовка",
    "вуз", "университет", "кафедра", "факультет", "часть", "объем", "час",
    "трудоемкость", "самостоятельный", "контроль", "оценка", "результат", "средство",
    "способ", "принцип", "подход", "характеристика", "свойство", "особенность",
    "процесс", "организация", "предприятие", "условие", "требование",
    "уровень", "этап", "вид", "тип", "форма", "ход", "поиск", "определение",
    "обеспечение", "проведение", "создание", "построение", "реализация",
    "основной", "основный", "общий", "различный", "современный", "новый", "сложный",
    "простой", "специальный", "практический", "теоретический", "прикладной",
    "локальный", "потенциальный", "некоторый", "определенный", "связанный",
    "технология", "краткий", "содержание", "методология", "методика", "вопрос",
    "аспект", "рассмотрение", "обзор", "сущность", "роль", "место", "значение",
    "состояние", "постановка", "раскрытие", "внутренний", "внешний", "рабочий",
    "документация", "упорядочивание", "преимущество", "критерий", "услуга",
    "развитие", "совершенствование", "внедрение", "эксплуатация", "сопровождение",
    "квалификация", "работник", "минобрнауки", "россия", "федерация", "единица",
    "зачетный", "высший", "высочайший", "образование",
}

# Универсальные абстрактные и процедурные существительные, которые не подходят
# на роль главного слова (вершины) предметного интереса в любой отрасли
PROCEDURAL_HEAD_NOUNS: set[str] = {
    "схема", "последовательность", "правило", "инструмент", "инструментарий",
    "номенклатура", "классификация", "событие", "документ", "функция",
    "воздействие", "ситуация", "блок", "компонент", "элемент", "представление",
    "изменение", "соединение", "написание", "состоятельность", "выбор",
    "обработка", "отчетность", "спецификация", "метод", "техника", "теория",
    "модель", "диаграмма", "достижение", "договоренность", "исследование",
}

# Универсальные самодостаточные одиночные термины (1-граммы)
ALLOWED_STANDALONE_TERMS: set[str] = {
    "блокчейн", "криптография", "кластеризация", "классификация", "регрессия",
    "инсоляция", "звукоизоляция", "теплотехника", "климатология", "светотехника",
    "акустика", "макетирование", "градостроительство", "менеджмент", "маркетинг",
    "логистика", "аутсорсинг", "факторинг", "лизинг", "инвестиции", "itsm", "itil",
    "sql", "nosql", "uml", "bpmn", "erp", "crm", "scada", "bim", "cad",
}

KNOWN_ABBREVIATIONS: set[str] = {
    "ИС", "БД", "СУБД", "ОС", "ЭВМ", "АСУ", "САПР", "ЖКХ", "АИС", "СМК", "ПГС", "1С",
}

ALL_STOPWORDS_NORM: set[str] = {
    w.replace("ё", "е") for w in (GENERAL_STOPWORDS | ACADEMIC_STOPWORDS)
}
PROCEDURAL_HEAD_NORM: set[str] = {
    w.replace("ё", "е") for w in PROCEDURAL_HEAD_NOUNS
}


def _norm_yo(word: str) -> str:
    """Нормализует букву 'ё' в 'е' для надёжных проверок по словарям."""
    return word.lower().replace("ё", "е")


@dataclass(frozen=True)
class TokenInfo:
    """Морфологический профиль отдельного слова."""

    surface: str       # Исходная форма слова в тексте
    lemma: str         # Начальная словарная форма (лемма)
    pos: str           # Часть речи (NOUN, ADJF, PRTF, LATN, ABBR)
    case: str | None   # Падеж (nomn, gent и т.д.)


class TextPreprocessor:
    """
    Универсальный двухпоточный лингвистический препроцессор текстов учебных дисциплин.
    """

    def __init__(self) -> None:
        self.morph = pymorphy3.MorphAnalyzer()

    def clean_for_dense(self, raw_text: str) -> str:
        """Поток 1 (Dense Stream): подготавливает связный текст для нейросети FRIDA."""
        if not raw_text:
            return ""

        text = re.sub(r"<[^>]+>", " ", raw_text)
        text = re.sub(
            r"(?i)(изучаемые объекты( дисциплины)?|краткое содержание( дисциплины)?|цели и задачи( дисциплины)?)\s*:?",
            ". ",
            text,
        )
        # Восстанавливаем частые слипшиеся в PDF сложные слова с дефисом
        text = re.sub(
            r"(?i)\bобъектноориентированн", "объектно-ориентированн", text
        )
        text = re.sub(
            r"(?i)\bматериальнопроизводственн", "материально-производственн", text
        )
        text = re.sub(
            r"(?i)\bдорожностроительн", "дорожно-строительн", text
        )
        text = re.sub(
            r"(?i)\bархитектурнохудожественн", "архитектурно-художественн", text
        )

        # Склеиваем слова, разорванные дефисом с пробелом/переносом строки ("ор- ганизации")
        text = re.sub(r"([а-яёА-ЯЁ]{2,})-\s+([а-яёА-ЯЁ]{2,})", r"\1\2", text)
        text = re.sub(r"[^\w\s\.,;:\-\(\)\+/]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _analyze_token(self, word: str) -> TokenInfo | None:
        """Производит морфологический анализ слова через PyMorphy3 с защитой аббревиатур."""
        word_clean = word.strip("-")
        if len(word_clean) < 2 or word_clean.isdigit():
            return None

        if word_clean.upper() in KNOWN_ABBREVIATIONS:
            abbr = word_clean.upper()
            return TokenInfo(surface=abbr, lemma=abbr, pos="ABBR", case="gent")

        word_low = word_clean.lower()

        if re.match(r"^[a-zA-Z0-9\-]+$", word_low):
            if _norm_yo(word_low) in ALL_STOPWORDS_NORM or len(word_low) < 2:
                return None
            return TokenInfo(
                surface=word_low,
                lemma=word_low,
                pos="LATN",
                case=None,
            )

        if _norm_yo(word_low) in ALL_STOPWORDS_NORM:
            return None

        parsed_variants = self.morph.parse(word_low)
        if not parsed_variants:
            return None

        # Если первый вариант разбора — глагол или причастие, но есть именной разбор NOUN,
        # выбираем существительное (исправляет омонимию вроде "насыпей" -> "насыпь")
        best = parsed_variants[0]
        if str(best.tag.POS) in ("VERB", "INFN", "GRND", "PRTF", "PRTS"):
            for alt in parsed_variants[1:]:
                if str(alt.tag.POS) == "NOUN":
                    best = alt
                    break

        lemma = str(best.normal_form)
        pos = str(best.tag.POS) if best.tag.POS else "UNKN"
        case = str(best.tag.case) if best.tag.case else None

        if word_low in ("данных", "данные", "данным", "данными"):
            lemma = "данные"
            pos = "NOUN"
            case = "gent" if word_low == "данных" else "nomn"

        if _norm_yo(lemma) in ALL_STOPWORDS_NORM:
            return None

        return TokenInfo(
            surface=word_low,
            lemma=lemma,
            pos=pos,
            case=case,
        )

    def _inflect_adj_to_noun(self, adj_token: TokenInfo, noun_lemma: str) -> str:
        """Согласует прилагательное или причастие в роде и числе с главным существительным."""
        if "-" in adj_token.surface:
            parts = adj_token.surface.split("-", maxsplit=1)
            second_token = TokenInfo(
                surface=parts[1], lemma=parts[1], pos=adj_token.pos, case=adj_token.case
            )
            inflected_second = self._inflect_adj_to_noun(second_token, noun_lemma)
            return f"{parts[0]}-{inflected_second}"

        noun_parses = self.morph.parse(noun_lemma)
        if not noun_parses:
            return adj_token.lemma

        noun_p = noun_parses[0]
        target_grammemes: set[str] = {"nomn"}

        if noun_lemma == "данные":
            target_grammemes.add("plur")
        else:
            if noun_p.tag.number:
                target_grammemes.add(str(noun_p.tag.number))
            if noun_p.tag.gender and str(noun_p.tag.number) == "sing":
                target_grammemes.add(str(noun_p.tag.gender))

        surface_parses = self.morph.parse(adj_token.surface)
        for p in surface_parses:
            if str(p.tag.POS) in ("ADJF", "PRTF"):
                inflected = p.inflect(target_grammemes)
                if inflected:
                    return str(inflected.word)

        return adj_token.lemma

    def _normalize_gent_dependent(self, token: TokenInfo) -> str:
        """Унифицирует форму зависимого существительного в родительном падеже."""
        if token.pos == "ABBR":
            return token.surface
        if token.lemma == "данные" or token.surface == "данных":
            return "данных"
        if "-" in token.surface:
            return token.surface

        parses = self.morph.parse(token.lemma)
        if not parses:
            return token.surface

        # Предпочитаем существительное при склонении зависимого слова
        p = next((v for v in parses if str(v.tag.POS) == "NOUN"), parses[0])
        plural_gent_words = {
            "последовательность", "отклонение", "операция", "система", "сеть",
            "база", "акция", "метод", "алгоритм", "конструкция", "сооружение",
            "здание", "материал", "фундамент", "основание", "проект", "риск",
            "ресурс", "услуга", "процесс", "интерфейс", "устройство", "приложение",
            "насыпь", "таблица", "знание",
        }
        if _norm_yo(token.lemma) in plural_gent_words:
            inflected = p.inflect({"plur", "gent"})
        else:
            inflected = p.inflect({"sing", "gent"})

        return str(inflected.word) if inflected else token.surface

    def extract_candidate_terms(
        self, raw_text: str, *, allow_procedural_head: bool = False
    ) -> Counter[str]:
        """
        Поток 2 (Sparse Stream): извлекает нормализованные терминологические словосочетания
        (2-3 слова) и строгий белый список одиночных научных терминов.
        """
        cleaned = self.clean_for_dense(raw_text)
        clauses = re.split(r"[\.,;:\(\)\n]+", cleaned)

        term_counter: Counter[str] = Counter()

        for clause in clauses:
            raw_words = re.findall(r"[а-яА-ЯёЁa-zA-Z0-9\-]+", clause)
            tokens: list[TokenInfo | None] = [self._analyze_token(w) for w in raw_words]
            n = len(tokens)

            i = 0
            while i < n:
                t1 = tokens[i]
                if t1 is None:
                    i += 1
                    continue

                # Шаблон 1: ADJF + NOUN + NOUN (в род. падеже)
                if (
                    i + 2 < n
                    and tokens[i + 1] is not None
                    and tokens[i + 2] is not None
                ):
                    t2 = tokens[i + 1]
                    t3 = tokens[i + 2]
                    assert t2 is not None and t3 is not None
                    if (
                        t1.pos in ("ADJF", "PRTF")
                        and t2.pos == "NOUN"
                        and (allow_procedural_head or _norm_yo(t2.lemma) not in PROCEDURAL_HEAD_NORM)
                        and t3.pos in ("NOUN", "ABBR")
                        and (t3.case == "gent" or t3.surface == "данных")
                    ):
                        adj_agreed = self._inflect_adj_to_noun(t1, t2.lemma)
                        dep_norm = self._normalize_gent_dependent(t3)
                        phrase = f"{adj_agreed} {t2.lemma} {dep_norm}"
                        term_counter[phrase] += 1
                        i += 3
                        continue

                # Шаблон 2: ADJF + NOUN
                if i + 1 < n and tokens[i + 1] is not None:
                    t2 = tokens[i + 1]
                    assert t2 is not None
                    if (
                        t1.pos in ("ADJF", "PRTF")
                        and t2.pos == "NOUN"
                        and (allow_procedural_head or _norm_yo(t2.lemma) not in PROCEDURAL_HEAD_NORM)
                    ):
                        adj_agreed = self._inflect_adj_to_noun(t1, t2.lemma)
                        noun_form = "данные" if t2.lemma == "данные" else t2.lemma
                        phrase = f"{adj_agreed} {noun_form}"
                        term_counter[phrase] += 1
                        i += 2
                        continue

                    # Шаблон 3: NOUN + NOUN в род. падеже
                    if (
                        t1.pos == "NOUN"
                        and (allow_procedural_head or _norm_yo(t1.lemma) not in PROCEDURAL_HEAD_NORM)
                        and t2.pos in ("NOUN", "ABBR")
                        and (t2.case == "gent" or t2.surface == "данных")
                    ):
                        if t1.lemma == "задача":
                            if t2.lemma in ALLOWED_STANDALONE_TERMS:
                                term_counter[t2.lemma] += 1
                        else:
                            dep_norm = self._normalize_gent_dependent(t2)
                            phrase = f"{t1.lemma} {dep_norm}"
                            term_counter[phrase] += 1
                        i += 2
                        continue

                    # Шаблон 4: Связки с англоязычными терминами
                    if (
                        (t1.pos == "LATN" and t2.pos in ("NOUN", "LATN"))
                        or (t1.pos == "NOUN" and t2.pos == "LATN")
                    ):
                        if allow_procedural_head or _norm_yo(t1.lemma) not in PROCEDURAL_HEAD_NORM:
                            phrase = f"{t1.lemma} {t2.lemma}"
                            term_counter[phrase] += 1
                        i += 2
                        continue

                # Шаблон 5: Одиночное слово из белого списка самодостаточных терминов
                if (
                    t1.lemma in ALLOWED_STANDALONE_TERMS
                    or t1.surface in ALLOWED_STANDALONE_TERMS
                ):
                    term_counter[t1.lemma] += 1

                i += 1

        return term_counter