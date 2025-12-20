from dataclasses import dataclass
from datetime import datetime

from src.models.enums import WordSource


@dataclass
class UserWord:
    source: WordSource
    collection_name: str
    lemma: str
    text: str
    translation_text: str
    translation_lemma: str
    creation_datetime: datetime
