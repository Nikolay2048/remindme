import csv
import json
import logging
import random
import string
from datetime import datetime, timezone

import requests

from src.models.enums import WordSource
from src.models.user_data import UserDataYandexTranslator
from src.models.word import UserWord
from src.services.word_processor import WordProcessor

logger = logging.getLogger(__name__)


class YandexTranslatorProcessor:
    """
    Обрабатывает информацию с Яндекс переводчика

    """

    @staticmethod
    def fetch_collection(collection_id: str) -> dict:
        yandexuid = "".join(random.choices(string.digits, k=18))

        url = f"https://translate.yandex.ru/props/api/collections/{collection_id}?srv=tr-text&uid"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": f"https://translate.yandex.ru/subscribe?collection_id={collection_id}",
        }
        cookies = {
            "first_visit_src": "collection_share_desktop",
            "yandexuid": yandexuid,
        }

        r = requests.get(url, headers=headers, cookies=cookies, timeout=30)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def extract_words(payload: dict):
        # Обычно структура такая: payload["collection"]["records"][...]
        records = payload.get("collection", {}).get("records", [])
        words = []

        collection_name = payload.get("collection", {}).get("name", "")
        for rec in records:
            if rec.get("lang") == "ru-en":
                dst = rec.get("text", "")
                src = rec.get("translation", "")
            elif rec.get("lang") == "en-ru":
                src = rec.get("text", "")
                dst = rec.get("translation", "")
            else:
                logger.error(f"Unsupported YandexTranslator language: {rec.get('lang')}")
                dst = src = ""
            wp = WordProcessor()
            if src or dst:
                word = UserWord(
                    source=WordSource.YANDEX_TRANSLATOR,
                    collection_name=collection_name,
                    lemma=" ".join(wp.get_lexemes(src)),
                    translation_lemma=" ".join(wp.get_lexemes(dst)),
                    text=src,
                    translation_text=dst,
                    creation_datetime=datetime.fromtimestamp(rec.get("creationTimestamp", ""), tz=timezone.utc),
                )
                words.append(word)
        return words



