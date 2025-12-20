from dataclasses import dataclass
from datetime import datetime

from src.models.enums import WordSource


# TODO: оформление словаря в БД

@dataclass
class LearningWord:
    source: WordSource
    collections: list[str]
    lemma: str
    text: str
    translation: str
    creation_datetime: datetime


class Lexeme:
    language: str
