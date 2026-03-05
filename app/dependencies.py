"""
FastAPI зависимости (Dependency Injection)
"""
import re
from typing import Annotated

from fastapi import Depends, HTTPException, status, Request

from .config import (
    SUPPORTED_TASK_TYPES,
    MODELS_DIR,
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_TRAIN_PER_HOUR,
)
from .models import Item
from .utils.logging_config import get_logger

logger = get_logger(__name__)


# Тип для аннотаций
TaskTypeDep = Annotated[str, Depends(lambda: None)]


def validate_model_name(name: str) -> str:
    """
    Валидировать имя модели (защита от path traversal)
    
    Args:
        name: Имя модели
        
    Returns:
        Валидированное имя
        
    Raises:
        HTTPException: Если имя некорректно
    """
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Имя модели не указано"
        )
    
    # Проверка на допустимые символы (только буквы, цифры, дефис, подчёркивание)
    if not re.match(r'^[\w\-]+$', name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Некорректное имя модели. Разрешены только буквы, цифры, дефис и подчёркивание"
        )
    
    # Проверка длины
    if len(name) > 64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Имя модели слишком длинное (максимум 64 символа)"
        )
    
    return name


def validate_task_type(task_type: str) -> str:
    """
    Валидировать тип задачи
    
    Args:
        task_type: Тип задачи
        
    Returns:
        Валидированный тип задачи
        
    Raises:
        HTTPException: Если тип не поддерживается
    """
    if task_type not in SUPPORTED_TASK_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Неподдерживаемый тип задачи: {task_type}. "
                f"Поддерживаемые: {', '.join(SUPPORTED_TASK_TYPES)}"
            )
        )
    return task_type


async def validate_item(item: Item) -> Item:
    """
    Валидировать Item модель (дополнительные проверки)
    
    Args:
        item: Item модель из запроса
        
    Returns:
        Валидированный Item
        
    Raises:
        HTTPException: Если валидация не пройдена
    """
    # Валидация имени модели
    validate_model_name(item.NameModel)
    
    # Валидация типа задачи
    validate_task_type(item.TaskType)
    
    # Проверка данных
    if not item.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Данные не предоставлены"
        )
    
    return item


async def check_rate_limit(request: Request) -> None:
    """
    Проверка rate limiting (заглушка для будущей реализации)
    
    Args:
        request: FastAPI request объект
        
    Note:
        Для полноценной реализации рекомендуется использовать slowapi
    """
    # TODO: Реализовать с использованием slowapi или redis
    pass


async def get_models_dir() -> str:
    """
    Получить директорию для хранения моделей
    
    Returns:
        Путь к директории моделей
    """
    return str(MODELS_DIR)


# Готовые зависимости для использования в endpoints
ModelNameDep = Annotated[str, Depends(validate_model_name)]
TaskTypeDep = Annotated[str, Depends(validate_task_type)]
ItemDep = Annotated[Item, Depends(validate_item)]
RateLimitDep = Annotated[None, Depends(check_rate_limit)]
