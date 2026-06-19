import asyncio
import joblib
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import psutil
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score

from app.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    MAX_MODEL_CACHE_SIZE,
    MODEL_TIMEOUT,
)


# ─────────────────────────────────────────────
# Кэш моделей
# ─────────────────────────────────────────────
class ModelCache:
    """Простой кэш моделей в памяти с ограничением по размеру."""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._metadata: Dict[str, Dict] = {}

    def get(self, name: str):
        return self._models.get(name)

    def put(self, name: str, model: Any, metadata: Dict) -> None:
        self._models[name] = model
        self._timestamps[name] = datetime.now()
        self._metadata[name] = metadata
        self._evict_if_needed()

    def touch(self, name: str) -> None:
        if name in self._timestamps:
            self._timestamps[name] = datetime.now()

    def get_metadata(self, name: str) -> Optional[Dict]:
        return self._metadata.get(name)

    def _evict_if_needed(self) -> None:
        if len(self._models) > MAX_MODEL_CACHE_SIZE:
            oldest_name = min(self._timestamps, key=self._timestamps.get)
            del self._models[oldest_name]
            del self._timestamps[oldest_name]
            del self._metadata[oldest_name]
            logging.info(f"Очищен кэш: удалена модель {oldest_name}")


# Глобальный экземпляр кэша
model_cache = ModelCache()

# ─────────────────────────────────────────────
# Эмбеддинги
# ─────────────────────────────────────────────
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logging.info(f"Загрузка модели эмбеддингов {EMBEDDING_MODEL_NAME}...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info("Модель эмбеддингов загружена")
    return _embedding_model


def generate_embeddings(texts) -> np.ndarray:
    model = get_embedding_model()
    clean_texts = [str(t) if pd.notna(t) else '' for t in texts]
    return model.encode(clean_texts, show_progress_bar=False)


def add_embeddings_to_df(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    logging.info(f"Генерация эмбеддингов для колонки '{text_column}'...")
    texts = df[text_column].tolist()
    embeddings = generate_embeddings(texts)
    emb_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(EMBEDDING_DIM)])
    result_df = pd.concat([df.drop(columns=[text_column]), emb_df], axis=1)
    logging.info(f"Эмбеддинги добавлены, форма: {result_df.shape}")
    return result_df


# ─────────────────────────────────────────────
# Метрики
# ─────────────────────────────────────────────
def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, np.argmax(y_pred, axis=1), average='macro')


# ─────────────────────────────────────────────
# Предобработка данных
# ─────────────────────────────────────────────
def _is_numeric_dtype(dtype) -> bool:
    try:
        return np.issubdtype(dtype, np.number)
    except (TypeError, AttributeError):
        if hasattr(dtype, 'kind'):
            return dtype.kind in ('i', 'u', 'f', 'c')
        return False


def _is_bool_dtype(dtype) -> bool:
    try:
        return np.issubdtype(dtype, np.bool_)
    except (TypeError, AttributeError):
        if hasattr(dtype, 'kind'):
            return dtype.kind == 'b'
        return False


def preprocess_data(metadata: Optional[Dict], new_data: pd.DataFrame) -> pd.DataFrame:
    if not metadata:
        return new_data

    template_df = pd.DataFrame(columns=metadata['columns'])

    for col, dtype in metadata['dtypes'].items():
        if col in template_df.columns:
            try:
                template_df[col] = template_df[col].astype(dtype)
            except Exception:
                pass

    aligned_df = pd.concat([template_df, new_data], ignore_index=True)

    fill_values = {}
    for col, dtype in metadata['dtypes'].items():
        if col in aligned_df.columns:
            if _is_numeric_dtype(dtype):
                fill_values[col] = 0
            elif _is_bool_dtype(dtype):
                fill_values[col] = False
            else:
                fill_values[col] = 'missing'

    aligned_df.fillna(fill_values, inplace=True)
    return aligned_df[metadata['columns']].reset_index(drop=True)


# ─────────────────────────────────────────────
# Вспомогательные
# ─────────────────────────────────────────────
def find_max_indices(arr, mapping):
    max_indices = []
    for sub_arr in arr:
        max_value = max(sub_arr)
        max_index = sub_arr.index(max_value)
        if mapping is not None and isinstance(mapping, dict) and len(mapping) > 0:
            keys = list(mapping.keys())
            if max_index < len(keys):
                max_indices.append(keys[max_index])
            else:
                max_indices.append(max_index)
        else:
            max_indices.append(max_index)
    return max_indices


# ─────────────────────────────────────────────
# Подготовка данных для обучения
# ─────────────────────────────────────────────
def normalize_text_column(text_column) -> Optional[str]:
    """Нормализует формат текстовой колонки (список или строка)."""
    if text_column is None:
        return None
    if isinstance(text_column, list):
        if len(text_column) == 0:
            return None
        if len(text_column) > 1:
            logging.warning(
                f"Несколько текстовых колонок не поддерживаются, используется первая: {text_column[0]}"
            )
        return text_column[0]
    return text_column


def ensure_model_dir(name: str) -> str:
    """Создаёт директорию модели и возвращает путь."""
    model_dir = f'./app/{name}'
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    return model_dir


def create_automl(task_type: str, df: pd.DataFrame):
    """Создаёт и обучает TabularAutoML."""
    from lightautoml.tasks import Task
    from lightautoml.automl.presets.tabular_presets import TabularAutoML

    roles = {'target': 'TARGET'}

    # Для классификации удаляем классы с одним примером
    if task_type in ('multiclass', 'binary'):
        value_counts = df['TARGET'].value_counts()
        values_to_drop = value_counts[value_counts == 1].index
        df = df[~df['TARGET'].isin(values_to_drop)]

        if len(df['TARGET'].unique()) < 2:
            raise ValueError("Недостаточно классов для обучения")

        task = Task(task_type, metric=f1_macro)
    else:
        task = Task(task_type)

    total_memory = psutil.virtual_memory().total / (1024**3)

    automl = TabularAutoML(
        task=task,
        cpu_limit=psutil.cpu_count(),
        timeout=MODEL_TIMEOUT,
        memory_limit=total_memory,
        general_params={'nested_cv': False, 'use_algos': 'auto'},
    )

    return automl, df, roles


async def save_model(automl, name: str, metadata: Dict) -> None:
    """Асинхронно сохраняет модель и метаданные."""
    model_path = f'./app/{name}/model.pkl'
    metadata_path = f'./app/{name}/metadata.pkl'

    await asyncio.to_thread(joblib.dump, automl, model_path)
    await asyncio.to_thread(joblib.dump, metadata, metadata_path)


async def load_model(name: str):
    """Загружает модель и метаданные из файла."""
    model_path = f'./app/{name}/model.pkl'
    metadata_path = f'./app/{name}/metadata.pkl'

    automl = await asyncio.to_thread(joblib.load, model_path)
    metadata = await asyncio.to_thread(joblib.load, metadata_path)
    return automl, metadata
