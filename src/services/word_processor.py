import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Set, Literal

import spacy
import pymorphy3

Lang = Literal["en", "ru"]

NLP_EN = spacy.load("en_core_web_sm")
MORPH_RU = pymorphy3.MorphAnalyzer()

RE_EN = re.compile(r"[A-Za-z]+")
RE_RU = re.compile(r"[А-Яа-яЁё]+")


def detect_lang(text: str) -> Optional[Lang]:
    """есть кириллица -> ru, если латиница -> en."""
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


class WordProcessor:
    def __init__(self, keep_stopwords: bool = False, min_len: int = 2):
        self.keep_stopwords = keep_stopwords
        self.min_len = min_len

    def get_lexemes(
        self,
        text: str,
        language: Optional[Lang] = None,
        *,
        as_set: bool = True
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
            out = []
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
        lexemes = self.get_lexemes(word, language=language, as_set=False)
        return lexemes[0] if lexemes else None

    def _lexemes_en(self, text: str) -> List[str]:
        doc = NLP_EN(text)
        res: List[str] = []
        for token in doc:
            if not token.is_alpha:
                continue
            if len(token.text) < self.min_len:
                continue
            if not self.keep_stopwords and token.is_stop:
                continue
            lemma = token.lemma_.lower().strip()
            if len(lemma) < self.min_len:
                continue
            # spaCy иногда отдаёт -PRON- (старые модели), на всякий случай:
            if lemma == "-pron-":
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
