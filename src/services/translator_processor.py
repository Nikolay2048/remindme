import logging
import random
import re
import string
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs

import requests

from src.db.repositories import UserVocabularyRepository
from src.models.enums import WordSource
from src.models.word import UserWord
from src.services.word_processor import WordProcessor

logger = logging.getLogger(__name__)

# Regex validator for Yandex Translator collection id
COLLECTION_ID_RE = re.compile(r"^[a-f0-9]{24}$")


class YandexTranslatorProcessor:
    """
    Обрабатывает информацию с Яндекс переводчика

    """

    def _fetch_collection(self, collection_id: str) -> dict:
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

    def _extract_words(self, payload: dict):
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

    def _extract_collection_id(self, value):
        value = value.strip()

        if COLLECTION_ID_RE.fullmatch(value):
            return value

        parsed = urlparse(value)
        query = parse_qs(parsed.query)
        collection_ids = query.get("collection_id")
        if collection_ids:
            cid = collection_ids[0]
            if COLLECTION_ID_RE.fullmatch(cid):
                return cid
        return None

    def process_user_collections(self, user_id: int, collections: list[str]):
        for collection in collections:
            collection_id = self._extract_collection_id(collection)
            if not collection_id:
                logger.error(f"Collection ID not valid: {collection}")
                raise ValueError(f"Collection id (or link) {collection} is not valid")
            logger.info(f"Collecting words from collection={collection_id}")
            payload = self._fetch_collection(collection_id)
            logger.debug(payload)
            words = self._extract_words(payload)
            logger.info(f"Collected words pairs: {len(words)}")
            UserVocabularyRepository.insert_or_update_word(user_id, words)
