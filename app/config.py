import logging
import numpy as np
from typing import Any, Optional, List, Union
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# Логирование
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    filename='log.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ─────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────
RANDOM_STATE = 45
MAX_MODEL_CACHE_SIZE = 10
MODEL_TIMEOUT = 14400  # 4 часа

np.random.seed(RANDOM_STATE)

EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384

# ─────────────────────────────────────────────
# Pydantic модели
# ─────────────────────────────────────────────
class Item(BaseModel):
    data: Any = Field(..., description="Данные для обучения или предсказания в формате JSON")
    NameModel: str = Field(..., description="Уникальное имя модели для сохранения/загрузки")
    TaskType: str = Field(..., description="Тип задачи (multiclass, binary, reg)")
    df_text: Optional[Any] = Field(None, description="Текстовая колонка для эмбеддингов")
    df_drop: Optional[Any] = Field(None, description="Колонки для удаления")


class PredictionResponse(BaseModel):
    predictions: List[Union[float, int, str]] = Field(..., description="Результаты предсказания")
