"""
LTAutoML API - Главный файл приложения

REST API для автоматического машинного обучения с использованием LightAutoML.
Поддерживает задачи классификации (бинарной и многоклассовой) и регрессии.
"""
import logging
from typing import List, Union

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import (
    APP_VERSION,
    APP_TITLE,
    APP_DESCRIPTION,
    HOST,
    PORT,
    MAX_MODEL_CACHE_SIZE,
    MODELS_DIR,
    TARGET_COLUMN,
)
from app.models import Item, HealthResponse
from app.utils import ModelCache, setup_logging, get_logger
from app.services import MLService
from app.dependencies import validate_item

# Настройка логирования
setup_logging()
logger = get_logger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    """Обработчик превышения rate limit"""
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Превышен лимит запросов. Попробуйте позже."
    )

app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Настройте для продакшена
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация сервисов
model_cache = ModelCache(max_size=MAX_MODEL_CACHE_SIZE)
ml_service = MLService(MODELS_DIR)

logger.info("LTAutoML API инициализирован")


@app.get("/", response_model=dict)
async def root():
    """
    Корневой endpoint - информация о API
    
    Returns:
        dict: Приветственное сообщение и доступные endpoints
    """
    return {
        "message": "Добро пожаловать в LTAutoML API!",
        "version": APP_VERSION,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "train": "POST /fit_predict/",
            "predict": "POST /predict/",
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Health check endpoint

    Returns:
        HealthResponse: Статус приложения
    """
    cache_size = await model_cache.size()
    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        models_cached=cache_size,
    )


@app.post("/fit_predict/", response_model=str)
async def fit_predict(request: Request, item: Item) -> str:
    """
    Обучить модель на предоставленных данных и сохранить
    
    Args:
        request: FastAPI request объект (для rate limiting)
        item: Данные для обучения
        
    Returns:
        str: Статус обучения
        
    Raises:
        HTTPException: При ошибках валидации или обучения
    """
    logger.info(f"Запрос на обучение модели: {item.NameModel}")
    
    # Валидация через dependency
    await validate_item(item)
    
    try:
        # Создание DataFrame
        df = pd.DataFrame(item.data)
    except Exception as e:
        logger.error(f"Ошибка при создании DataFrame: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ошибка при создании DataFrame: {e}"
        )
    
    # Проверка наличия целевой переменной
    if TARGET_COLUMN not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Колонка '{TARGET_COLUMN}' не найдена в данных"
        )
    
    # Подготовка ролей
    roles = {"target": TARGET_COLUMN}
    if item.df_text is not None:
        roles["text"] = item.df_text
    if item.df_drop is not None:
        roles["drop"] = item.df_drop
    
    try:
        # Обучение модели
        metadata = await ml_service.train(
            df=df,
            model_name=item.NameModel,
            task_type=item.TaskType,
            roles=roles,
        )
        
        # Сохранение в кэш
        model_path = MODELS_DIR / item.NameModel / "model.pkl"
        model, _ = await ml_service.load_model(item.NameModel)
        await model_cache.set(item.NameModel, model, metadata)
        
        logger.info(f"Обучение модели {item.NameModel} завершено успешно")
        return "Обучение завершено"
        
    except FileNotFoundError as e:
        logger.error(f"Ошибка пути к файлу: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка сохранения модели: {e}"
        )
    except ValueError as e:
        logger.error(f"Ошибка валидации данных: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при обучении: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обучении модели: {e}"
        )


@app.post("/predict/", response_model=List[Union[float, int, str, list]])
async def predict(request: Request, item: Item) -> List[Union[float, int, str]]:
    """
    Выполнить предсказание с использованием обученной модели
    
    Args:
        request: FastAPI request объект (для rate limiting)
        item: Данные для предсказания
        
    Returns:
        List[Union[float, int, str]]: Результаты предсказания
        
    Raises:
        HTTPException: При ошибках валидации или предсказания
    """
    logger.info(f"Запрос на предсказание модели: {item.NameModel}")
    
    # Валидация через dependency
    await validate_item(item)
    
    try:
        # Создание DataFrame
        df = pd.DataFrame(item.data)
    except Exception as e:
        logger.error(f"Ошибка при создании DataFrame: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ошибка при создании DataFrame: {e}"
        )
    
    # Проверка кэша
    model = await model_cache.get(item.NameModel)
    metadata = await model_cache.get_metadata(item.NameModel)
    
    if model is None:
        logger.info(f"Модель {item.NameModel} не найдена в кэше, загрузка из файла")
        try:
            model, metadata = await ml_service.load_model(item.NameModel)
            await model_cache.set(item.NameModel, model, metadata)
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Модель не найдена: {item.NameModel}"
            )
    
    try:
        # Предсказание
        predictions = await ml_service.predict(
            df=df,
            model_name=item.NameModel,
            task_type=item.TaskType,
            model=model,
            metadata=metadata,
        )
        
        logger.info(f"Предсказание для {item.NameModel} завершено")
        return predictions
        
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при предсказании: {e}"
        )


if __name__ == "__main__":
    logger.info(f"Запуск сервера на {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
