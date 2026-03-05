"""
Сервис для работы с ML моделями (LightAutoML)
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import asyncio
import joblib
import numpy as np
import pandas as pd
import psutil

from lightautoml.tasks import Task
from lightautoml.automl.presets.text_presets import TabularNLPAutoML
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from sklearn.metrics import f1_score

from ..config import (
    RANDOM_STATE,
    MODEL_TIMEOUT,
    TARGET_COLUMN,
    MIN_CLASS_SAMPLES,
    BERT_MODEL_NAME,
    BERT_LANGUAGE,
    DEFAULT_ALGORITHMS,
    CV_FOLDS,
    NESTED_CV,
    CPU_COUNT,
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет F1 macro метрику

    Args:
        y_true: Истинные значения
        y_pred: Предсказанные значения (probabilities или predictions)

    Returns:
        F1 macro метрика
    """
    # Для бинарной классификации y_pred может быть 1D
    if y_pred.ndim == 1:
        y_pred_classes = (y_pred > 0.5).astype(int)
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)
    
    return f1_score(y_true, y_pred_classes, average='macro')


class MLService:
    """
    Сервис для обучения и предсказания ML моделей
    """

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._memory_limit_gb = psutil.virtual_memory().total / (1024**3)

    def _get_model_paths(self, model_name: str) -> tuple[Path, Path]:
        """
        Получить пути к файлам модели и метаданных
        
        Args:
            model_name: Имя модели
            
        Returns:
            Кортеж (путь к модели, путь к метаданным)
        """
        model_dir = self.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return (
            model_dir / "model.pkl",
            model_dir / "metadata.pkl"
        )

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Проверить наличие целевой переменной
        
        Args:
            df: DataFrame с данными
            
        Raises:
            ValueError: Если TARGET колонка не найдена
        """
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Колонка '{TARGET_COLUMN}' не найдена в данных")

    def _filter_rare_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Удалить редкие классы из данных
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с отфильтрованными редкими классами
        """
        value_counts = df[TARGET_COLUMN].value_counts()
        values_to_drop = value_counts[value_counts <= MIN_CLASS_SAMPLES].index
        if len(values_to_drop) > 0:
            logger.info(f"Удалены редкие классы: {list(values_to_drop)}")
            df = df[~df[TARGET_COLUMN].isin(values_to_drop)]
        return df

    def _validate_classes(self, df: pd.DataFrame) -> None:
        """
        Проверить минимальное количество классов
        
        Args:
            df: DataFrame с данными
            
        Raises:
            ValueError: Если классов меньше 2
        """
        unique_classes = df[TARGET_COLUMN].nunique()
        if unique_classes < 2:
            raise ValueError(
                f"Недостаточно классов для обучения: {unique_classes} < 2"
            )

    def _create_task(
        self,
        task_type: str,
        is_classification: bool = True
    ) -> Task:
        """
        Создать задачу LightAutoML
        
        Args:
            task_type: Тип задачи (multiclass, binary, reg)
            is_classification: Флаг классификации (для метрики)
            
        Returns:
            Task объект
        """
        if is_classification:
            return Task(task_type, metric=f1_macro)
        return Task(task_type)

    def _create_automl(
        self,
        task: Task,
        has_text: bool = False
    ) -> Union[TabularNLPAutoML, TabularAutoML]:
        """
        Создать AutoML модель
        
        Args:
            task: Task объект
            has_text: Флаг наличия текстовых данных
            
        Returns:
            AutoML модель
        """
        common_params = {
            "task": task,
            "timeout": MODEL_TIMEOUT,
            "cpu_limit": CPU_COUNT,
            "memory_limit": self._memory_limit_gb,
            "reader_params": {
                "n_jobs": CPU_COUNT,
                "cv": CV_FOLDS,
                "random_state": RANDOM_STATE,
            },
            "general_params": {
                "nested_cv": NESTED_CV,
                "use_algos": DEFAULT_ALGORITHMS,
            },
        }

        if has_text:
            return TabularNLPAutoML(
                **common_params,
                text_params={
                    "lang": BERT_LANGUAGE,
                    "bert_model": BERT_MODEL_NAME,
                },
            )
        return TabularAutoML(**common_params)

    async def train(
        self,
        df: pd.DataFrame,
        model_name: str,
        task_type: str,
        roles: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Обучить модель и сохранить
        
        Args:
            df: DataFrame с данными для обучения
            model_name: Имя модели
            task_type: Тип задачи
            roles: Роли колонок для AutoML
            
        Returns:
            Метаданные обученной модели
        """
        logger.info(f"Начало обучения модели {model_name}")

        # Валидация данных
        self._validate_data(df)

        # Фильтрация редких классов для классификации
        if task_type in ("multiclass", "binary"):
            df = self._filter_rare_classes(df)
            self._validate_classes(df)

        # Создание задачи и модели
        is_classification = task_type in ("multiclass", "binary")
        task = self._create_task(task_type, is_classification)
        has_text = "text" in roles
        automl = self._create_automl(task, has_text)

        # Обучение модели
        logger.info(f"Запуск fit_predict для {model_name}")
        await asyncio.to_thread(
            automl.fit_predict, df, roles=roles, verbose=2
        )

        # Сохранение модели
        model_path, metadata_path = self._get_model_paths(model_name)
        
        logger.info(f"Сохранение модели в {model_path}")
        await asyncio.to_thread(joblib.dump, automl, model_path)

        # Сохранение метаданных
        metadata = {
            "columns": df.columns.tolist(),
            "dtypes": {
                col: str(dtype) for col, dtype in df.dtypes.items()
            },
            "task_type": task_type,
            "trained_at": datetime.now().isoformat(),
        }
        
        logger.info(f"Сохранение метаданных в {metadata_path}")
        await asyncio.to_thread(joblib.dump, metadata, metadata_path)

        logger.info(f"Обучение модели {model_name} завершено")

        return metadata

    async def predict(
        self,
        df: pd.DataFrame,
        model_name: str,
        task_type: str,
        model: Any,
        metadata: Dict[str, Any],
    ) -> List[Union[float, int, str]]:
        """
        Выполнить предсказание
        
        Args:
            df: DataFrame с данными для предсказания
            model_name: Имя модели
            task_type: Тип задачи
            model: Загруженная ML модель
            metadata: Метаданные модели
            
        Returns:
            Список предсказаний
        """
        logger.info(f"Выполнение предсказания для модели {model_name}")

        # Предобработка данных
        new_df = self.preprocess_data(metadata, df)

        # Предсказание
        logger.info("Вызов model.predict")
        test_pred = await asyncio.to_thread(model.predict, new_df)

        # Обработка результатов
        if task_type in ("multiclass", "binary"):
            return self._process_classification_predictions(
                test_pred.data.tolist(), model
            )
        return test_pred.data.tolist()

    def _process_classification_predictions(
        self,
        predictions: List[List[float]],
        model: Any
    ) -> List[Union[int, str]]:
        """
        Обработать предсказания классификации
        
        Args:
            predictions: Сырые предсказания модели
            model: ML модель
            
        Returns:
            Список классов
        """
        # Проверка class_mapping
        if hasattr(model, 'reader') and model.reader is not None:
            class_mapping = getattr(model.reader, 'class_mapping', None)
            if class_mapping and isinstance(class_mapping, dict):
                return self._map_predictions(predictions, class_mapping)
        
        # Возврат индексов классов
        return [np.argmax(row) for row in predictions]

    def _map_predictions(
        self,
        predictions: List[List[float]],
        mapping: Dict[str, Any]
    ) -> List[Union[str, int]]:
        """
        Преобразовать индексы в названия классов
        
        Args:
            predictions: Сырые предсказания
            mapping: Словарь сопоставления классов
            
        Returns:
            Список названий классов
        """
        keys = list(mapping.keys())
        result = []
        
        for sub_arr in predictions:
            max_index = int(np.argmax(sub_arr))
            if max_index < len(keys):
                result.append(keys[max_index])
            else:
                result.append(max_index)
        
        return result

    @staticmethod
    def preprocess_data(
        metadata: Dict[str, Any],
        new_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Предобработать новые данные для соответствия обученной модели
        
        Args:
            metadata: Метаданные обученной модели
            new_data: Новые данные для предсказания
            
        Returns:
            Предобработанный DataFrame
        """
        if not metadata:
            return new_data

        columns = metadata.get("columns", [])
        dtypes = metadata.get("dtypes", {})

        if not columns:
            return new_data

        # Создаём шаблон с нужными колонками
        template_df = pd.DataFrame(columns=columns)

        # Применяем типы данных
        for col, dtype_str in dtypes.items():
            if col in template_df.columns:
                try:
                    # Восстанавливаем тип из строкового представления
                    if 'int' in dtype_str:
                        template_df[col] = template_df[col].astype('Int64')
                    elif 'float' in dtype_str:
                        template_df[col] = template_df[col].astype('float64')
                    elif 'bool' in dtype_str:
                        template_df[col] = template_df[col].astype('bool')
                    elif 'datetime' in dtype_str:
                        template_df[col] = template_df[col].astype('datetime64[ns]')
                    else:
                        template_df[col] = template_df[col].astype('object')
                except Exception:
                    template_df[col] = template_df[col].astype('object')

        # Объединяем с новыми данными
        aligned_df = pd.concat([template_df, new_data], ignore_index=True)

        # Заполняем пропуски
        fill_values = {}
        for col, dtype_str in dtypes.items():
            if col in aligned_df.columns:
                if 'int' in dtype_str or 'float' in dtype_str:
                    fill_values[col] = 0
                elif 'bool' in dtype_str:
                    fill_values[col] = False
                else:
                    fill_values[col] = 'missing'

        aligned_df = aligned_df.fillna(fill_values)

        # Возвращаем только нужные колонки
        return aligned_df[columns].reset_index(drop=True)

    async def load_model(self, model_name: str) -> tuple[Any, Dict[str, Any]]:
        """
        Загрузить модель и метаданные из файла
        
        Args:
            model_name: Имя модели
            
        Returns:
            Кортеж (модель, метаданные)
            
        Raises:
            FileNotFoundError: Если модель не найдена
        """
        model_path, metadata_path = self._get_model_paths(model_name)

        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Метаданные не найдены: {metadata_path}")

        logger.info(f"Загрузка модели из {model_path}")
        model = await asyncio.to_thread(joblib.load, model_path)
        
        logger.info(f"Загрузка метаданных из {metadata_path}")
        metadata = await asyncio.to_thread(joblib.load, metadata_path)

        return model, metadata
