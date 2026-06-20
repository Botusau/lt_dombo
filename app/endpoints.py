import asyncio
import logging
import traceback
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.config import Item
from app.helpers import (
    add_embeddings_to_df,
    create_automl,
    ensure_model_dir,
    find_max_indices,
    load_model,
    model_cache,
    normalize_text_column,
    preprocess_data,
    save_model,
)

router = APIRouter()


@router.get("/")
async def home_page() -> dict:
    return {"message": "Добро пожаловать в LTAutoML API!"}


@router.post("/fit_predict/")
async def fit_predict(item: Item) -> str:
    logging.info(f"Запуск обучения модели {item.NameModel}")

    # ── Валидация ──────────────────────────────────────
    if not item.data:
        raise HTTPException(status_code=400, detail="Данные не предоставлены")
    if not item.NameModel:
        raise HTTPException(status_code=400, detail="Имя модели не указано")
    if item.TaskType not in ('multiclass', 'binary', 'reg'):
        raise HTTPException(status_code=400, detail=f"Неподдерживаемый тип задачи: {item.TaskType}")

    try:
        df = pd.DataFrame(item.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при создании DataFrame: {e}")

    if 'TARGET' not in df.columns:
        raise HTTPException(status_code=400, detail="Колонка TARGET не найдена в данных")

    # ── Подготовка директории ──────────────────────────
    ensure_model_dir(item.NameModel)

    original_columns = df.columns.tolist()
    original_dtypes = df.dtypes.to_dict()

    # ── Эмбеддинги ─────────────────────────────────────
    text_column = normalize_text_column(item.df_text)
    if text_column is not None:
        if text_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Текстовая колонка '{text_column}' не найдена")
        df = add_embeddings_to_df(df, text_column)

    # ── Обучение ───────────────────────────────────────
    try:
        automl, df, roles = create_automl(item.TaskType, df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if item.df_drop is not None:
        roles['drop'] = item.df_drop

    try:
        await asyncio.to_thread(automl.fit_predict, df, roles=roles, verbose=10, log_file="fit.log")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обучении модели: {e}")

    # ── Сохранение ─────────────────────────────────────
    metadata = {
        'columns': original_columns,
        'dtypes': original_dtypes,
        'text_column': text_column,
    }

    try:
        await save_model(automl, item.NameModel, metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении: {e}")

    # ── Кэш ────────────────────────────────────────────
    model_cache.put(item.NameModel, automl, metadata)

    logging.info("Обучение завершено")
    return 'Обучение завершено'


@router.post("/predict/")
async def predict(item: Item) -> List[Union[float, int, str, list]]:
    logging.info(f"Запуск предсказания модели {item.NameModel}")

    try:
        # ── Валидация ──────────────────────────────────
        if not item.data:
            raise HTTPException(status_code=400, detail="Данные не предоставлены")
        if not item.NameModel:
            raise HTTPException(status_code=400, detail="Имя модели не указано")
        if item.TaskType not in ('multiclass', 'binary', 'reg'):
            raise HTTPException(status_code=400, detail=f"Неподдерживаемый тип задачи: {item.TaskType}")

        try:
            df = pd.DataFrame(item.data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Ошибка при создании DataFrame: {e}")

        model_path = Path(f'/data/models/{item.NameModel}/model.pkl')
        if not model_path.exists():
            raise HTTPException(status_code=400, detail=f"Файл модели не найден: {model_path}")

        # ── Загрузка / кэш ─────────────────────────────
        automl = model_cache.get(item.NameModel)
        metadata = model_cache.get_metadata(item.NameModel)

        if automl is None:
            logging.info(f"Загрузка модели из {model_path}")
            automl, metadata = await load_model(item.NameModel)
            model_cache.put(item.NameModel, automl, metadata)
        else:
            model_cache.touch(item.NameModel)

        # ── Предобработка ──────────────────────────────
        new_df = preprocess_data(metadata, df)

        text_column = (metadata or {}).get('text_column')
        if text_column is not None and text_column in new_df.columns:
            new_df = add_embeddings_to_df(new_df, text_column)

        # ── Предсказание ───────────────────────────────
        test_pred = await asyncio.to_thread(automl.predict, new_df)

        if item.TaskType in ('multiclass', 'binary'):
            if hasattr(automl.reader, 'class_mapping') and automl.reader.class_mapping is not None:
                return find_max_indices(test_pred.data.tolist(), automl.reader.class_mapping)
            return [np.argmax(row) for row in test_pred.data.tolist()]
        return test_pred.data.tolist()

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Необработанная ошибка в /predict/: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Необработанная ошибка: {e}")
