from src.models.user_data import UserDataYandexTranslator


class YandexTranslatorProcessor:
    """
    Обрабатывает информацию с Яндекс переводчика

    """

    def __init__(self, user_data: UserDataYandexTranslator) -> None:
        self._words_collections = user_data.collections

    def collect_words_from_collections(self):
        for collection in self._words_collections:
            pass
