"""
Pydantic модели для запросов и ответов API
"""
from typing import Any, List, Union

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .config import SUPPORTED_TASK_TYPES, MAX_DATA_SIZE


class Item(BaseModel):
    """
    Модель данных для запросов API обучения и предсказания

    Attributes:
        data: Данные для обучения или предсказания в формате JSON
        NameModel: Уникальное имя модели для сохранения/загрузки
        TaskType: Тип задачи машинного обучения (multiclass, binary, reg)
        df_text: Опциональное поле для указания текстовой колонки
        df_drop: Опциональное поле для указания колонок, которые нужно удалить
    """
    model_config = ConfigDict(extra='forbid')

    data: Any = Field(
        ...,
        description="Данные для обучения или предсказания в формате JSON",
        examples=[[{"feature1": 1, "feature2": 2, "TARGET": "class1"}]]
    )
    NameModel: str = Field(
        ...,
        description="Уникальное имя модели для сохранения/загрузки",
        min_length=1,
        max_length=64,
        pattern=r'^[\w\-]+$',
        examples=["my_model"]
    )
    TaskType: str = Field(
        ...,
        description="Тип задачи машинного обучения (multiclass, binary, reg)",
        examples=["multiclass"]
    )
    df_text: Union[Any, None] = Field(
        None,
        description="Опциональное поле для указания текстовой колонки",
        examples=["text_column"]
    )
    df_drop: Union[Any, None] = Field(
        None,
        description="Опциональное поле для указания колонок, которые нужно удалить",
        examples=[["drop_column1", "drop_column2"]]
    )

    @field_validator('data')
    @classmethod
    def validate_data_size(cls, v: Any) -> Any:
        """Проверка размера данных"""
        if isinstance(v, list) and len(v) > MAX_DATA_SIZE:
            raise ValueError(
                f"Превышен максимальный размер данных: {len(v)} > {MAX_DATA_SIZE}"
            )
        return v

    @field_validator('TaskType')
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        """Проверка типа задачи"""
        if v not in SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"Неподдерживаемый тип задачи: {v}. "
                f"Поддерживаемые: {', '.join(SUPPORTED_TASK_TYPES)}"
            )
        return v


class PredictionResponse(BaseModel):
    """
    Модель ответа для предсказаний

    Attributes:
        predictions: Результаты предсказания модели
    """
    model_config = ConfigDict(extra='forbid')

    predictions: List[Union[float, int, str]] = Field(
        ...,
        description="Результаты предсказания модели"
    )


class HealthResponse(BaseModel):
    """
    Модель ответа для health check endpoint

    Attributes:
        status: Статус приложения
        version: Версия приложения
        models_cached: Количество закэшированных моделей
    """
    model_config = ConfigDict(extra='forbid')

    status: str = Field(..., description="Статус приложения")
    version: str = Field(..., description="Версия приложения")
    models_cached: int = Field(..., description="Количество закэшированных моделей")


class ErrorResponse(BaseModel):
    """
    Модель ответа для ошибок

    Attributes:
        detail: Описание ошибки
    """
    model_config = ConfigDict(extra='forbid')

    detail: str = Field(..., description="Описание ошибки")
