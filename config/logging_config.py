import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "default": {
            "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        },
        "short": {
            "format": "%(levelname)s | %(message)s"
        }
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG",
            "stream": "ext://sys.stdout"
        },
        "file_app": {
            "class": "logging.FileHandler",
            "filename": os.path.join(LOG_DIR, "app.log"),
            "formatter": "default",
            "level": "DEBUG"
        },
        "file_network": {
            "class": "logging.FileHandler",
            "filename": os.path.join(LOG_DIR, "network.log"),
            "formatter": "default",
            "level": "DEBUG"
        },
        "file_providers": {
            "class": "logging.FileHandler",
            "filename": os.path.join(LOG_DIR, "data_providers.log"),
            "formatter": "default",
            "level": "DEBUG"
        },

        "file_features_processing": {
            "class": "logging.FileHandler",
            "filename": os.path.join(LOG_DIR, "features_processing.log"),
            "formatter": "default",
            "level": "DEBUG"
        },
        "out_of_handler": {
            "class": "logging.FileHandler",
            "filename": os.path.join(LOG_DIR, "log.log"),
            "formatter": "default",
            "level": "DEBUG"
        },
    },

    "loggers": {
        "src.utils.requests_helper": {
            "handlers": ["console", "file_network"],
            "level": "DEBUG",
            "propagate": False
        },
        "urllib3": {
            "handlers": ["console", "file_network"],
            "level": "DEBUG",
            "propagate": False
        },
        # "services.api": {
        #     "handlers": ["console", "file_errors"],
        #     "level": "INFO",
        #     "propagate": False
        # },
        "src.services": {
            "handlers": ["console", "file_providers"],
            "level": "DEBUG",
            "propagate": False
        },
        "src.features": {
            "handlers": ["console", "file_features_processing"],
            "level": "DEBUG",
            "propagate": False
        },
        "__main__": {
            "handlers": ["console", "file_app"],
            "level": "DEBUG",
            "propagate": False
        },

        # root logger по умолчанию
        "": {
            "handlers": ["console", "out_of_handler"],
            "level": "DEBUG"
        }
    }
}
