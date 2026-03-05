"""
Настройка логирования с ротацией файлов
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ..config import LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOG_FILE


def setup_logging(
    log_file: Path = LOG_FILE,
    level: str = LOG_LEVEL,
    max_bytes: int = LOG_MAX_BYTES,
    backup_count: int = LOG_BACKUP_COUNT
) -> None:
    """
    Настроить логирование с ротацией файлов
    
    Args:
        log_file: Путь к файлу логов
        level: Уровень логирования
        max_bytes: Максимальный размер файла логов перед ротацией
        backup_count: Количество резервных файлов логов
    """
    # Создаём logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # Очищаем существующие handlers
    logger.handlers.clear()

    # Форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # RotatingFileHandler для файла
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # ConsoleHandler для вывода в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Логирование предупреждений о зависимостях
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('lightautoml').setLevel(logging.INFO)

    logger.info(f"Логирование настроено: файл={log_file}, уровень={level}")


def get_logger(name: str) -> logging.Logger:
    """
    Получить именованный logger
    
    Args:
        name: Имя logger (обычно __name__ модуля)
        
    Returns:
        Настроенный logger
    """
    return logging.getLogger(name)
