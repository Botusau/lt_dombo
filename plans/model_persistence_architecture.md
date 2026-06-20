# Архитектурное решение: Сохранение обученных моделей при перезапуске контейнера

## 1. Анализ текущей архитектуры

### Расположение моделей
На основе анализа кода:

| Артефакт | Путь (в контейнере) | Источник |
|---|---|---|
| Обученная модель | `./app/{NameModel}/model.pkl` | [`save_model()`](../app/helpers.py:227) |
| Метаданные модели | `./app/{NameModel}/metadata.pkl` | [`save_model()`](../app/helpers.py:227) |
| Кэш эмбеддингов | `/root/.cache/torch/sentence_transformers/` | [`get_embedding_model()`](../app/helpers.py:68) |

### Проблема
[`starter.sh`](../starter.sh:5) выполняет `rm -rf lt_dombo`, что удаляет:
- Весь каталог проекта
- Каталоги с обученными моделями (`app/{NameModel}/`)
- Кэш NLP-моделей (если хранится внутри проекта)

### Текущая конфигурация Docker
- **Dockerfile**: Минимальный, без volumes
- **docker-compose.yml**: Нет mounted volumes, используется готовый образ `botusau/lt_dombo:v1.0.0`

---

## 2. Варианты решения

### Вариант A: Docker Volume для каталога моделей ⭐ Рекомендованный

**Суть**: Подключить внешний Docker volume к каталогу моделей внутри контейнера.

**Схема**:
```
Host:/data/models  <-->  Container:/app/lt_dombo/app/models
```

**Плюсы**:
- Минимальные изменения в коде
- Стандартный подход Docker
- Модели сохраняются на хосте
- Легко бэкапить и монтировать

**Минусы**:
- Требует изменения путей в коде (из `./app/{name}` в `./app/models/{name}`)
- Нужно изменить `docker-compose.yml`

---

### Вариант B: Вынесение моделей за пределы клонируемого каталога

**Суть**: Сохранять модели в каталог `/data/models` внутри контейнера, который не удаляется скриптом.

**Схема**:
```
Container:/data/models/{NameModel}/model.pkl
```

**Плюсы**:
- Не требует изменения docker-compose
- Модели выживают после перезапуска

**Минусы**:
- Данные теряются при удалении контейнера (без volume)
- Нужно изменить пути в коде
- Неочевидная архитектура

---

### Вариант C: Комбинированный (Volume + вынесение)

**Суть**: Модели сохраняются в `/data/models` внутри контейнера + этот каталог подключается как volume на хост.

**Схема**:
```
Host:/data/models  <-->  Container:/data/models
```

**Плюсы**:
- Максимальная надёжность
- Модели не зависят от контейнера
- Не зависят от каталога проекта
- Легко масштабировать

**Минусы**:
- Требует изменений в коде и конфигурации Docker

---

## 3. Рекомендуемое решение: Вариант C

### Обоснование

| Критерий | Оценка |
|---|---|
| Простота реализации | ⭐⭐⭐ Средняя (изменения в 3 файлах) |
| Надёжность | ⭐⭐⭐⭐⭐ Максимальная |
| Совместимость | ⭐⭐⭐⭐ Полная совместимость |
| Масштабируемость | ⭐⭐⭐⭐⭐ Готово к росту |

### Почему не Вариант A?
- Модели остаются внутри каталога `app/`, что запутывает структуру
- При `rm -rf lt_dombo` теряется структура каталогов

### Почему Вариант C?
1. **Полная независимость**: Модели хранятся отдельно от кода
2. **Выживание перезапусков**: Volume сохраняется при любом обновлении
3. **Чистая архитектура**: Разделение кода и данных
4. **Бонус**: Можно также подключить volume для кэша NLP-моделей

---

## 4. Диаграмма архитектуры

```mermaid
flowchart TB
    subgraph Host[Хост-машина]
        V1[Volume: models_data]
        V2[Volume: nlp_cache_data]
    end

    subgraph Container[Docker-контейнер]
        subgraph AppDir[/app/lt_dombo]
            Code[Код приложения]
            Starter[starter.sh]
        end

        subgraph DataDir[/data]
            Models[/data/models]
            NLP_Cache[/data/nlp_cache]
        end
    end

    Git[(GitHub)] -->|git clone| Starter
    Starter -->|удаляет| AppDir
    Starter -->|НЕ затрагивает| DataDir

    V1 <-->|mount| Models
    V2 <-->|mount| NLP_Cache

    Code -->|сохраняет| Models
    Code -->|кэширует| NLP_Cache

    style Models fill:#90EE90
    style NLP_Cache fill:#90EE90
    style V1 fill:#87CEEB
    style V2 fill:#87CEEB
    style AppDir fill:#FFB6C1
```

---

## 5. Шаги реализации

### Шаг 1: Изменить [`app/helpers.py`](../app/helpers.py)

Изменить константы путей для моделей:

```python
# Добавить в начало файла
MODEL_STORAGE_DIR = '/data/models'

def ensure_model_dir(name: str) -> str:
    model_dir = f'{MODEL_STORAGE_DIR}/{name}'
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    return model_dir

async def save_model(automl, name: str, metadata: Dict) -> None:
    model_path = f'{MODEL_STORAGE_DIR}/{name}/model.pkl'
    metadata_path = f'{MODEL_STORAGE_DIR}/{name}/metadata.pkl'
    await asyncio.to_thread(joblib.dump, automl, model_path)
    await asyncio.to_thread(joblib.dump, metadata, metadata_path)

async def load_model(name: str):
    model_path = f'{MODEL_STORAGE_DIR}/{name}/model.pkl'
    metadata_path = f'{MODEL_STORAGE_DIR}/{name}/metadata.pkl'
    automl = await asyncio.to_thread(joblib.load, model_path)
    metadata = await asyncio.to_thread(joblib.load, metadata_path)
    return automl, metadata
```

### Шаг 2: Изменить [`app/endpoints.py`](../app/endpoints.py)

Исправить проверку существования модели:

```python
from pathlib import Path

# Заменить:
model_path = Path(f'./app/{item.NameModel}/model.pkl')
# На:
model_path = Path(f'/data/models/{item.NameModel}/model.pkl')
```

### Шаг 3: Изменить [`Dockerfile`](../Dockerfile)

Добавить создание каталогов данных:

```dockerfile
FROM python:3.12

# Создаём каталоги для данных
RUN mkdir -p /data/models /data/nlp_cache

EXPOSE 8000
COPY ./starter.sh ./app/starter.sh
WORKDIR /app
CMD ["./app/starter.sh"]
```

### Шаг 4: Изменить [`docker-compose.yml`](../docker-compose.yml)

Добавить volumes:

```yaml
version: '3.8'
services:
  lt_dombo:
    image: botusau/lt_dombo:v1.0.0
    restart: always
    build:
      context: .
      dockerfile: ./Dockerfile
    ports:
      - 8000:8000
    volumes:
      - models_data:/data/models
      - nlp_cache_data:/data/nlp_cache
    environment:
      - TRANSFORMERS_CACHE=/data/nlp_cache
      - SENTENCE_TRANSFORMERS_HOME=/data/nlp_cache

volumes:
  models_data:
    driver: local
  nlp_cache_data:
    driver: local
```

### Шаг 5: Изменить [`starter.sh`](../starter.sh)

Добавить создание каталогов на случай ручного запуска:

```bash
#!/bin/bash

# Создаём каталоги данных
mkdir -p /data/models
mkdir -p /data/nlp_cache

cd /app
rm -rf lt_dombo

git clone https://github.com/Botusau/lt_dombo.git

cd ./lt_dombo

sh run.sh
```

---

## 6. Проверка после реализации

После развёртывания:

1. **Обучить модель**: `POST /fit_predict/` с `NameModel="test_model"`
2. **Проверить сохранение**: Файл должен появиться в `/data/models/test_model/model.pkl`
3. **Перезапустить контейнер**: `docker compose restart`
4. **Проверить предсказание**: `POST /predict/` с тем же `NameModel` — модель должна найтись

---

## 7. Альтернатива для быстрого внедрения

Если изменения в коде нежелательны, можно использовать **только docker-compose volumes** для конкретного каталога:

```yaml
volumes:
  - models_data:/app/lt_dombo/app
```

**Но это НЕ сработает**, потому что `rm -rf lt_dombo` удаляет весь каталог, и volume будет пуст после клонирования.

Поэтому **обязательно** нужно менять пути в коде на `/data/models`.

---

## 8. Итоговая сводка изменений

| Файл | Изменение |
|---|---|
| [`app/helpers.py`](../app/helpers.py) | Изменить пути в `ensure_model_dir`, `save_model`, `load_model` |
| [`app/endpoints.py`](../app/endpoints.py) | Изменить путь в проверке `model_path.exists()` |
| [`Dockerfile`](../Dockerfile) | Добавить `RUN mkdir -p /data/models /data/nlp_cache` |
| [`docker-compose.yml`](../docker-compose.yml) | Добавить volumes и environment variables |
| [`starter.sh`](../starter.sh) | Добавить `mkdir -p /data/models /data/nlp_cache` |
