"""
Асинхронное кэширование моделей с блокировками
"""
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Асинхронный кэш для ML моделей с LRU eviction
    
    Attributes:
        max_size: Максимальное количество моделей в кэше
    """

    def __init__(self, max_size: int = 10):
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._max_size = max_size

    async def get(self, key: str) -> Optional[Any]:
        """
        Получить модель из кэша
        
        Args:
            key: Ключ модели (имя модели)
            
        Returns:
            Модель или None, если не найдена
        """
        async with self._lock:
            if key in self._cache:
                # Обновляем timestamp при доступе
                self._timestamps[key] = datetime.now()
                logger.debug(f"Модель {key} найдена в кэше")
                return self._cache[key]
            logger.debug(f"Модель {key} не найдена в кэше")
            return None

    async def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Получить метаданные модели из кэша"""
        async with self._lock:
            return self._metadata.get(key)

    async def set(
        self,
        key: str,
        model: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Сохранить модель в кэш
        
        Args:
            key: Ключ модели (имя модели)
            model: Модель для сохранения
            metadata: Опциональные метаданные модели
        """
        async with self._lock:
            self._cache[key] = model
            self._timestamps[key] = datetime.now()
            if metadata:
                self._metadata[key] = metadata
            
            logger.info(f"Модель {key} сохранена в кэш")
            
            # Очистка кэша при необходимости
            await self._evict_if_needed()

    async def _evict_if_needed(self) -> None:
        """Удалить самую старую модель, если кэш переполнен"""
        while len(self._cache) > self._max_size:
            # Находим самую старую модель
            oldest_key = min(
                self._timestamps.items(),
                key=lambda x: x[1]
            )[0]
            
            # Удаляем модель и её метаданные
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]
            if oldest_key in self._metadata:
                del self._metadata[oldest_key]
            
            logger.info(f"Очищен кэш: удалена модель {oldest_key}")

    async def delete(self, key: str) -> bool:
        """
        Удалить модель из кэша
        
        Args:
            key: Ключ модели
            
        Returns:
            True если модель была удалена, False если не найдена
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
                if key in self._metadata:
                    del self._metadata[key]
                logger.info(f"Модель {key} удалена из кэша")
                return True
            return False

    async def contains(self, key: str) -> bool:
        """Проверить наличие модели в кэше"""
        async with self._lock:
            return key in self._cache

    async def size(self) -> int:
        """Получить текущий размер кэша"""
        async with self._lock:
            return len(self._cache)

    async def clear(self) -> None:
        """Очистить весь кэш"""
        async with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._metadata.clear()
            logger.info("Кэш моделей очищен")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Получить статистику кэша
        
        Returns:
            Словарь со статистикой
        """
        async with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "models": list(self._cache.keys()),
                "timestamps": {
                    k: v.isoformat()
                    for k, v in self._timestamps.items()
                }
            }
