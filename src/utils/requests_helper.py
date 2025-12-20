import logging
import time
from typing import Optional

import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

logger = logging.getLogger(__name__)


def safe_get_response(url: str, retries: int = 3, delay: int = 2, timeout: int = 5) -> Optional[requests.Response]:
    """
    Пытается скачать содержимое по URL. При неудаче повторяет `retries` раз с задержкой `delay` секунд.

    :param url: URL для скачивания
    :param retries: Количество повторных попыток
    :param delay: Задержка между попытками в секундах
    :param timeout: Таймаут запроса в секундах
    :return: объект Response или None, если все попытки неудачны
    """

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except (ConnectionError, Timeout) as e:
            logger.warning(f"Попытка {attempt}/{retries} — сетевая ошибка: {e}")
            pass
        except RequestException as e:
            logger.warning(f"Повтор через {delay} секунд")
            break
        if attempt < retries:
            logger.debug(f"Повтор через {delay} секунд")
            time.sleep(delay)
    return None
