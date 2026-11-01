import re
from collections import Counter
from dataclasses import dataclass

import pymorphy3

# Общеязыковые стоп-слова и служебные частицы
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
}

# Базовые структурные слова академических документов (РПД)
ACADEMIC_STOPWORDS: set[str] = {
    "студент", "обучающийся", "дисциплина", "семестр", "курс", "лекция",
    "практика", "лабораторный", "работа", "занятие", "зачет", "экзамен",
    "аттестация", "компетенция", "индикатор", "знать", "уметь", "владеть",
    "навык", "цель", "объект", "предмет", "раздел", "тема", "модуль",
    "изучение", "освоение", "формирование", "применение", "использование",
    "выполнение", "решение", "основа", "введение", "понятие", "область",
    "способность", "готовность", "деятельность", "профессиональный", "учебный",
    "образовательный", "программа", "направление", "подготовка", "вуз",
    "университет", "кафедра", "факультет", "часть", "объем", "час", "трудоемкость",
    "самостоятельный", "контроль", "оценка", "результат", "средство",
    "способ", "принцип", "подход", "характеристика", "свойство", "особенность",
    "процесс", "организация", "предприятие", "условие", "требование",
    "уровень", "этап", "вид", "тип", "форма", "ход", "поиск", "определение",
    "обеспечение", "проведение", "создание", "построение", "реализация",
    "основной", "общий", "различный", "современный", "новый", "сложный",
    "простой", "специальный", "практический", "теоретический", "прикладной",
    "локальный", "потенциальный", "некоторый", "определенный", "связанный",
}

# Широкие существительные, которые НЕ могут быть самостоятельным интересом из 1 слова,
# но МОГУТ входить в составные словосочетания ("метод классификации", "база данных")
GENERIC_STANDALONE_NOUNS: set[str] = {
    "технология", "система", "данные", "информация", "анализ", "база",
    "модель", "алгоритм", "программа", "разработка", "проектирование",
    "управление", "среда", "структура", "элемент", "параметр", "значение",
    "множество", "класс", "категория", "шаблон", "правило", "сеть",
    "устройство", "оборудование", "комплекс", "инструмент", "пакет",
    "приложение", "интерфейс", "язык", "код", "функция", "таблица",
    "запрос", "сервер", "клиент", "пользователь", "сайт", "страница",
    "товар", "акция", "цена", "заемщик", "эксперимент", "ошибка", "ввод",
    "метод", "задача", "операция", "появление", "осознание", "факт",
    "знание", "звено", "предсказание", "деление", "фильтрация", "переход",
}

ALL_STOPWORDS: set[str] = GENERAL_STOPWORDS | ACADEMIC_STOPWORDS


@dataclass(frozen=True)
class TokenInfo:
    """Морфологический профиль отдельного слова."""

    surface: str       # Исходная форма слова в тексте (в нижнем регистре)
    lemma: str         # Начальная словарная форма (лемма)
    pos: str           # Часть речи (NOUN, ADJF, LATN и т.д.)
    case: str | None   # Падеж (nomn, gent и т.д.)


class TextPreprocessor:
    """
    Двухпоточный лингвистический препроцессор текстов учебных дисциплин.
    """

    def __init__(self) -> None:
        self.morph = pymorphy3.MorphAnalyzer()

    def clean_for_dense(self, raw_text: str) -> str:
        """
        Поток 1 (Dense Stream): подготавливает связный текст для нейросети FRIDA.
        """
        if not raw_text:
            return ""

        text = re.sub(r"<[^>]+>", " ", raw_text)
        text = re.sub(
            r"(?i)(изучаемые объекты( дисциплины)?|краткое содержание( дисциплины)?|цели и задачи( дисциплины)?)\s*:",
            ". ",
            text,
        )
        text = re.sub(r"([а-яёА-ЯЁ]{2,})-\s+([а-яёА-ЯЁ]{2,})", r"\1\2", text)
        text = re.sub(r"[^\w\s\.,;:\-\(\)\+/]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _analyze_token(self, word: str) -> TokenInfo | None:
        """Производит морфологический анализ слова через PyMorphy3."""
        word_low = word.lower().strip("-")
        if len(word_low) < 2 or word_low.isdigit():
            return None

        if re.match(r"^[a-zA-Z0-9\-]+$", word_low):
            if word_low in ALL_STOPWORDS:
                return None
            return TokenInfo(
                surface=word_low,
                lemma=word_low,
                pos="LATN",
                case=None,
            )

        parsed_variants = self.morph.parse(word_low)
        if not parsed_variants:
            return None

        best = parsed_variants[0]
        lemma = str(best.normal_form)
        pos = str(best.tag.POS) if best.tag.POS else "UNKN"
        case = str(best.tag.case) if best.tag.case else None

        if word_low in ("данных", "данные", "данным", "данными"):
            lemma = "данные"
            pos = "NOUN"
            case = "gent" if word_low == "данных" else "nomn"

        # Слово "задача" в сочетаниях "задача классификации" заменяем на пропуск,
        # но если за ним идет термин в род. падеже ("классификации"), переводим сам термин в им. падеж!
        if lemma in ALL_STOPWORDS or word_low in ALL_STOPWORDS:
            return None

        return TokenInfo(
            surface=word_low,
            lemma=lemma,
            pos=pos,
            case=case,
        )

    def _inflect_adj_to_noun(self, adj_word: str, noun_lemma: str) -> str:
        """Согласует прилагательное в роде и числе с главным существительным в им. падеже."""
        if noun_lemma == "данные":
            adj_parses = self.morph.parse(adj_word)
            if adj_parses:
                inflected = adj_parses[0].inflect({"plur", "nomn"})
                if inflected:
                    return str(inflected.word)

        adj_parses = self.morph.parse(adj_word)
        noun_parses = self.morph.parse(noun_lemma)
        if not adj_parses or not noun_parses:
            return adj_word

        adj_p = adj_parses[0]
        noun_p = noun_parses[0]

        target_grammemes: set[str] = {"nomn"}
        if noun_p.tag.number:
            target_grammemes.add(str(noun_p.tag.number))
        if noun_p.tag.gender and str(noun_p.tag.number) == "sing":
            target_grammemes.add(str(noun_p.tag.gender))

        inflected = adj_p.inflect(target_grammemes)
        return str(inflected.word) if inflected else str(adj_p.normal_form)

    def _normalize_gent_dependent(self, token: TokenInfo) -> str:
        """
        Унифицирует форму зависимого существительного в родительном падеже,
        чтобы 'анализ последовательности' и 'анализ последовательностей' не дублировались.
        """
        if token.lemma == "данные" or token.surface == "данных":
            return "данных"

        parses = self.morph.parse(token.lemma)
        if not parses:
            return token.surface

        p = parses[0]
        # Ставим во множественное число род. падежа для исчисляемых объектов ("последовательностей", "отклонений", "акций")
        # либо в единственное для неисчисляемых ("информации", "регрессии", "классификации")
        if token.lemma in {"последовательность", "отклонение", "операция", "система", "сеть", "база", "акция", "метод", "алгоритм"}:
            inflected = p.inflect({"plur", "gent"})
        else:
            inflected = p.inflect({"sing", "gent"})

        return str(inflected.word) if inflected else token.surface

    def extract_candidate_terms(self, raw_text: str) -> Counter[str]:
        """
        Поток 2 (Sparse Stream): извлекает нормализованные терминологические словосочетания.
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

                # Шаблон 1: ADJF + NOUN + NOUN (в род. падеже) -> "интеллектуальный анализ данных"
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
                        and t3.pos == "NOUN"
                        and (t3.case == "gent" or t3.surface == "данных")
                    ):
                        adj_agreed = self._inflect_adj_to_noun(t1.lemma, t2.lemma)
                        dep_norm = self._normalize_gent_dependent(t3)
                        phrase = f"{adj_agreed} {t2.lemma} {dep_norm}"
                        term_counter[phrase] += 1
                        i += 3
                        continue

                # Шаблон 2: ADJF + NOUN -> "нейронная сеть", "сиквенциальный анализ"
                if i + 1 < n and tokens[i + 1] is not None:
                    t2 = tokens[i + 1]
                    assert t2 is not None
                    if t1.pos in ("ADJF", "PRTF") and t2.pos == "NOUN":
                        adj_agreed = self._inflect_adj_to_noun(t1.lemma, t2.lemma)
                        noun_form = "данные" if t2.lemma == "данные" else t2.lemma
                        phrase = f"{adj_agreed} {noun_form}"
                        term_counter[phrase] += 1
                        i += 2
                        continue

                    # Шаблон 3: NOUN + NOUN в род. падеже -> "база данных", "анализ последовательностей"
                    if (
                        t1.pos == "NOUN"
                        and t2.pos == "NOUN"
                        and (t2.case == "gent" or t2.surface == "данных")
                    ):
                        # Если первое слово — это "задача" (напр., "задача классификации", "задача регрессии"),
                        # то извлекаем сам предмет задачи в именительном падеже ("классификация", "регрессия")!
                        if t1.lemma == "задача":
                            if t2.lemma not in GENERIC_STANDALONE_NOUNS:
                                term_counter[t2.lemma] += 1
                        else:
                            dep_norm = self._normalize_gent_dependent(t2)
                            phrase = f"{t1.lemma} {dep_norm}"
                            term_counter[phrase] += 1
                        i += 2
                        continue

                    # Шаблон 4: Связка с англоязычным термином (LATN + NOUN или NOUN + LATN)
                    if (t1.pos == "LATN" and t2.pos == "NOUN") or (
                        t1.pos == "NOUN" and t2.pos == "LATN"
                    ):
                        phrase = f"{t1.lemma} {t2.lemma}"
                        term_counter[phrase] += 1
                        i += 2
                        continue

                # Шаблон 5: Одиночное содержательное существительное
                if (
                    t1.pos in ("NOUN", "LATN")
                    and len(t1.lemma) >= 4
                    and t1.lemma not in GENERIC_STANDALONE_NOUNS
                ):
                    term_counter[t1.lemma] += 1

                i += 1

        return term_counter