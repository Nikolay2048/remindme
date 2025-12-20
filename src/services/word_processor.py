import re
from functools import lru_cache
from typing import Iterable, List, Optional, Set, Literal, Dict

import spacy
import pymorphy3

Lang = Literal["en", "ru"]

NLP_EN = spacy.load("en_core_web_sm", disable=["parser", "ner"])
MORPH_RU = pymorphy3.MorphAnalyzer()

RE_EN = re.compile(r"[A-Za-z]+")
RE_RU = re.compile(r"[А-Яа-яЁё]+")


def detect_lang(text: str) -> Optional[Lang]:
    """есть кириллица -> ru, если латиница -> en. Смешанный текст -> None."""
    text = text or ""
    has_ru = bool(RE_RU.search(text))
    has_en = bool(RE_EN.search(text))
    if has_ru and not has_en:
        return "ru"
    if has_en and not has_ru:
        return "en"
    if has_ru and has_en:
        return None
    return None


@lru_cache(maxsize=200_000)
def lemmatize_ru_word(word: str) -> str:
    return MORPH_RU.parse(word)[0].normal_form


@lru_cache(maxsize=500_000)
def lemmatize_en_word(word: str) -> str:
    """
    Быстрая лемматизация одного английского слова с кэшем.
    Важно: вызываем spaCy на ОДНО слово, а не на длинную строку.
    """
    w = (word or "").strip().lower()
    if not w:
        return w
    if not RE_EN.fullmatch(w):
        # если вдруг прилетела пунктуация/цифры
        return w

    doc = NLP_EN(w)
    for t in doc:
        if t.is_alpha:
            lemma = (t.lemma_ or "").lower().strip()
            if lemma == "-pron-" or not lemma:
                return t.text.lower()
            return lemma
    return w


class WordProcessor:
    def __init__(self, keep_stopwords: bool = False, min_len: int = 2):
        self.keep_stopwords = keep_stopwords
        self.min_len = min_len

    def get_lexemes(
        self,
        text: str,
        language: Optional[Lang] = None,
        *,
        as_set: bool = True,
    ) -> Set[str] | List[str]:
        """
        Универсальный метод:
        - если text = слово -> вернёт лемму
        - если text = фраза/субтитры -> вернёт леммы всех слов
        """
        text = (text or "").strip()
        if not text:
            return set() if as_set else []

        lang = language or detect_lang(text)
        if lang is None:
            en_part = " ".join(RE_EN.findall(text))
            ru_part = " ".join(RE_RU.findall(text))
            out: List[str] = []
            if en_part:
                out.extend(self._lexemes_en(en_part))
            if ru_part:
                out.extend(self._lexemes_ru(ru_part))
            return set(out) if as_set else out

        if lang == "en":
            out = self._lexemes_en(text)
        elif lang == "ru":
            out = self._lexemes_ru(text)
        else:
            raise ValueError(f"Unsupported language: {lang}")

        return set(out) if as_set else out

    def get_lexeme(self, word: str, language: Optional[Lang] = None) -> Optional[str]:
        """
        Лемма слова.
        """
        word = (word or "").strip()
        if not word:
            return None

        lang = language or detect_lang(word)

        if lang == "en":
            lemma = lemmatize_en_word(word)
            if len(lemma) < self.min_len:
                return None
            if (not self.keep_stopwords) and lemma in {"a", "an", "the"}:
                return None
            return lemma

        if lang == "ru":
            w = word.lower()
            if len(w) < self.min_len:
                return None
            lemma = lemmatize_ru_word(w)
            return lemma if len(lemma) >= self.min_len else None

        # mixed / неизвестно: fallback через общий парсер
        lexemes = self.get_lexemes(word, language=lang, as_set=False)
        return lexemes[0] if lexemes else None

    def get_lexeme_map(
        self,
        words: Iterable[str],
        *,
        language: Lang = "en",
        batch_size: int = 2048,
    ) -> Dict[str, str]:
        """

        """
        if language != "en":
            # для RU проще дергать pymorphy3 по одному (но он и так быстрый + кэш)
            out: Dict[str, str] = {}
            for w in words:
                raw = (w or "").strip().lower()
                if not raw or len(raw) < self.min_len:
                    continue
                if not RE_RU.fullmatch(raw):
                    continue
                out[raw] = lemmatize_ru_word(raw)
            return out

        cleaned: List[str] = []
        for w in words:
            raw = (w or "").strip().lower()
            if not raw:
                continue
            if len(raw) < self.min_len:
                continue
            if not RE_EN.fullmatch(raw):
                continue
            cleaned.append(raw)

        # чтобы не гонять одно и то же
        cleaned = list(dict.fromkeys(cleaned))

        out: Dict[str, str] = {}

        # nlp.pipe гораздо быстрее чем 10000 раз NLP_EN(word)
        for doc in NLP_EN.pipe(cleaned, batch_size=batch_size):
            # doc соответствует одному элементу cleaned
            raw = doc.text.lower().strip()
            lemma = raw
            for t in doc:
                if t.is_alpha:
                    cand = (t.lemma_ or "").lower().strip()
                    if cand and cand != "-pron-":
                        lemma = cand
                    else:
                        lemma = t.text.lower()
                    break

            if len(lemma) < self.min_len:
                continue

            # стоп-слова на уровне леммы (опционально)
            if (not self.keep_stopwords) and lemma in {"a", "an", "the"}:
                continue

            out[raw] = lemma

        return out

    # --- internal helpers ---

    def _lexemes_en(self, text: str) -> List[str]:
        """

        """
        doc = NLP_EN(text)
        res: List[str] = []
        for token in doc:
            if not token.is_alpha:
                continue
            if len(token.text) < self.min_len:
                continue
            if not self.keep_stopwords and token.is_stop:
                continue
            lemma = (token.lemma_ or "").lower().strip()
            if not lemma or lemma == "-pron-":
                continue
            if len(lemma) < self.min_len:
                continue
            res.append(lemma)
        return res

    def _lexemes_ru(self, text: str) -> List[str]:
        words = RE_RU.findall(text)
        res: List[str] = []
        for w in words:
            w = w.lower()
            if len(w) < self.min_len:
                continue
            lemma = lemmatize_ru_word(w)
            if len(lemma) < self.min_len:
                continue
            res.append(lemma)
        return res
