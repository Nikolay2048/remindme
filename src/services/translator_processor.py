import csv
import json
import logging
import random
import string

import requests

from src.models.user_data import UserDataYandexTranslator

logger = logging.getLogger(__name__)


class YandexTranslatorProcessor:
    """
    Обрабатывает информацию с Яндекс переводчика

    """

    def __init__(self, user_data: UserDataYandexTranslator) -> None:
        self._words_collections = user_data.collections

    def fetch_collection(self, collection_id: str) -> dict:
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

    def extract_pairs(self, payload: dict):
        # Обычно структура такая: payload["collection"]["records"][...]
        records = payload.get("collection", {}).get("records", [])
        pairs = []
        for rec in records:
            src = rec.get("text", "")
            dst = rec.get("translation", "")
            if src or dst:
                pairs.append((src, dst))
        return pairs

    def collect_words_from_collections(self):
        for collection_id in self._words_collections:
            logger.info(f"Collecting words from collection={collection_id}")
            payload = self.fetch_collection(collection_id)
            logger.debug(payload)
            pairs = self.extract_pairs(payload)
            logger.info(f"Collected words pairs: {len(pairs)}")

            with open("yandex_collection.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["text", "translation"])
                w.writerows(pairs)

            # если нужно — можно сохранить сырой JSON
            with open("yandex_collection_raw.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
