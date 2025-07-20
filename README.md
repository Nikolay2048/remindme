<pre lang="plaintext">
word_memory_project/
│
├── data/                    # Хранилище "сырых" и обработанных данных
│   ├── raw/                 # История просмотров, текст песен, субтитры и т.п.
│   ├── processed/           # Очистка, парсинг, извлечённые слова
│   └── external/            # Словари, частотные словари, embeddings и т.п.

├── notebooks/               # Jupyter ноутбуки для исследования
│   ├── EDA.ipynb            # Исследовательский анализ данных
│   ├── Embedding_Analysis.ipynb
│   └── Word_Matching.ipynb

├── src/                     # Исходный код проекта
│   ├── data/                # Модули для загрузки и предобработки
│   │   └── preprocessing.py
│   ├── features/            # Извлечение признаков и контекстов
│   │   └── embeddings.py
│   ├── models/              # ML/нейросетевые модели
│   │   └── association_model.py
│   ├── matching/            # Поиск совпадений слов в личной истории
│   │   └── match_words.py
│   └── utils/               # Вспомогательные функции (логгеры, конфиги)

├── config/                  # Конфигурационные файлы (YAML/JSON)
│   └── settings.yaml

├── tests/                   # Unit-тесты для компонентов
│   └── test_matching.py

├── app/                     # Интерфейс/обёртка
│   ├── api.py               # REST API или FastAPI/Flask-сервер
│   └── ui/                  # Веб-интерфейс или CLI

├── requirements.txt         # Зависимости
├── README.md
└── setup.py                 # Для установки как пакета (опционально)
</pre>