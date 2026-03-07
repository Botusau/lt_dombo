"""
Конфигурация приложения и константы
"""
import os
from pathlib import Path
from typing import Final


# Версия приложения
APP_VERSION: Final[str] = "1.0.0"

# FastAPI настройки
APP_TITLE: Final[str] = "LTAutoML API"
APP_DESCRIPTION: Final[str] = (
    "API для автоматического машинного обучения с использованием LightAutoML. "
    "Поддерживает задачи классификации и регрессии."
)
API_V1_PREFIX: Final[str] = "/api/v1"

# Сервер
HOST: Final[str] = "0.0.0.0"
PORT: Final[int] = 8000

# Random state для воспроизводимости
RANDOM_STATE: Final[int] = 45

# Кэширование моделей
MAX_MODEL_CACHE_SIZE: Final[int] = 10
MODEL_TIMEOUT: Final[int] = 14400  # 4 часа в секундах

# Пути
BASE_DIR: Final[Path] = Path(__file__).parent.parent
MODELS_DIR: Final[Path] = BASE_DIR / "app"
LOG_FILE: Final[Path] = BASE_DIR / "log.log"

# Логирование
LOG_MAX_BYTES: Final[int] = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: Final[int] = 5
LOG_LEVEL: Final[str] = "INFO"

# Ограничения
MAX_DATA_SIZE: Final[int] = 10_000_000  # Максимум строк в данных
MAX_REQUEST_SIZE: Final[int] = 50 * 1024 * 1024  # 50 MB макс размер запроса

# Rate limiting
RATE_LIMIT_PER_MINUTE: Final[int] = 5
RATE_LIMIT_TRAIN_PER_HOUR: Final[int] = 3

# Поддерживаемые типы задач
SUPPORTED_TASK_TYPES: Final[tuple[str, ...]] = ("multiclass", "binary", "reg")

# BERT модель для текстовых задач
BERT_MODEL_NAME: Final[str] = "DeepPavlov/rubert-base-cased-conversational"
BERT_LANGUAGE: Final[str] = "ru"

# Алгоритмы AutoML
#["linear_l2", "lgb", "cb", "nn", "lgb_tuned", "cb_tuned"]
DEFAULT_ALGORITHMS: Final[list[list[str]]] = [
    ["lgb", "cb", "lgb_tuned", "cb_tuned"]
]

# CV настройки
CV_FOLDS: Final[int] = 5
NESTED_CV: Final[bool] = False

# Имя целевой колонки
TARGET_COLUMN: Final[str] = "TARGET"

# Минимальное количество примеров класса для обучения
MIN_CLASS_SAMPLES: Final[int] = 4

# CPU лимиты (по умолчанию все доступные)
CPU_COUNT: Final[int] = os.cpu_count() or 4
