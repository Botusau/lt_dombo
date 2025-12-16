import os
from typing import Any, Optional, List, Union
import logging
import psutil
from pathlib import Path
from datetime import datetime
import asyncio
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import pandas as pd
import numpy as np

from lightautoml.tasks import Task
from lightautoml.automl.presets.text_presets import TabularNLPAutoML
from lightautoml.automl.presets.tabular_presets import TabularAutoML

import joblib
from sklearn.metrics import f1_score

# Настройка логирования
logging.basicConfig(level=logging.INFO, filename='log.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация FastAPI приложения
app = FastAPI(
    title="LTAutoML API",
    description="API для автоматического машинного обучения с использованием LightAutoML. Поддерживает задачи классификации и регрессии."
)

# Хранилище моделей в памяти
app.state.models = {}
# Хранилище времени последнего использования моделей
app.state.model_timestamps = {}
# Хранилище метаданных моделей
app.state.model_metadata = {}

# Константы
RANDOM_STATE = 45  # fixed random state for various reasons
MAX_MODEL_CACHE_SIZE = 10  # Максимальное количество моделей в кэше
MODEL_TIMEOUT = 14400  # Таймаут обучения в секундах (4 часа)

# Установка случайного состояния
np.random.seed(RANDOM_STATE)

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
  data: Any = Field(..., description="Данные для обучения или предсказания в формате JSON")
  NameModel: str = Field(..., description="Уникальное имя модели для сохранения/загрузки")
  TaskType: str = Field(..., description="Тип задачи машинного обучения (multiclass, binary, reg)")
  df_text: Optional[Any] = Field(None, description="Опциональное поле для указания текстовой колонки")
  df_drop: Optional[Any] = Field(None, description="Опциональное поле для указания колонок, которые нужно удалить")

class PredictionResponse(BaseModel):
  """
  Модель ответа для предсказаний
  
  Attributes:
      predictions: Результаты предсказания модели
  """
  predictions: List[Union[float, int, str]] = Field(..., description="Результаты предсказания модели")

def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """
  Вычисляет F1 макро метрику
  
  Args:
      y_true: Истинные значения
      y_pred: Предсказанные значения
      
  Returns:
      F1 макро метрика
  """
  return f1_score(y_true, np.argmax(y_pred, axis=1), average='macro')

@app.get("/", response_model=dict)
async def home_page():
  """
  Возвращает приветственное сообщение
  
  Returns:
      dict: Приветственное сообщение
  """
  return {"message": "Добро пожаловать в LTAutoML API!"}

@app.post("/fit_predict/", response_model=str)
async def fit_predict(item: Item) -> str:
  """
  Обучает модель на предоставленных данных и сохраняет её
  
  Args:
      item: Данные для обучения
      
  Returns:
      str: Статус обучения
      
  Raises:
      HTTPException: При ошибках валидации или обучения
  """
  logging.info(f"Запуск обучения модели {item.NameModel}")

  # Валидация входных данных
  if not item.data:
    logging.error("Ошибка: данные не предоставлены")
    raise HTTPException(status_code=400, detail="Данные не предоставлены")
  
  if not item.NameModel:
    logging.error("Ошибка: имя модели не указано")
    raise HTTPException(status_code=400, detail="Имя модели не указано")
  
  if item.TaskType not in ['multiclass', 'binary', 'reg']:
    logging.error(f"Ошибка: неподдерживаемый тип задачи: {item.TaskType}")
    raise HTTPException(status_code=400, detail=f"Неподдерживаемый тип задачи: {item.TaskType}")

  try:
    df = pd.DataFrame(item.data)
  except Exception as e:
    logging.error(f"Ошибка при создании DataFrame: {str(e)}")
    raise HTTPException(status_code=400, detail=f"Ошибка при создании DataFrame: {str(e)}")

  # Проверка наличия целевой переменной
  if 'TARGET' not in df.columns:
    logging.error("Ошибка: колонка TARGET не найдена в данных")
    raise HTTPException(status_code=400, detail="Колонка TARGET не найдена в данных")

  # Проверяем что директория существует если нет, то создаем
  if not Path('./app').exists():
    try:
      Path('./app').mkdir(parents=True, exist_ok=True)
    except Exception as e:
      logging.error(f"Ошибка: не удалось создать директорию /app: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Не удалось создать директорию /app: {str(e)}")

  NameModel = item.NameModel
  # Получаем путь к директории модели
  model_dir = './app/' + NameModel
  # Создаем директорию модели если она не существует
  if not Path(model_dir).exists():
    try:
      Path(model_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
      logging.error(f"Ошибка: не удалось создать директорию {model_dir}: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Не удалось создать директорию {model_dir}: {str(e)}")
    
  # Получаем путь к файлу модели
  patchModel = model_dir + '/model.pkl'
  # Получаем путь к файлу метаданным
  patchMetadata = model_dir + '/metadata.pkl'
  
  roles = {
        'target': 'TARGET',
    }
  
  if item.df_text is not None:
    roles['text'] = item.df_text

  if item.df_drop is not None:
    roles['drop'] = item.df_drop

  total_memory = psutil.virtual_memory().total / (1024**3)

  if item.TaskType == 'multiclass' or item.TaskType == 'binary':

    value_counts = df['TARGET'].value_counts() # Подсчитываем количество повторений каждого значения в колонке TARGET
    values_to_drop = value_counts[value_counts == 1].index # Находим значения, которые встречаются только один раз
    df = df[~df['TARGET'].isin(values_to_drop)] # Удаляем строки, где значение в колонке TARGET встречается только один раз

    # Проверка минимального количества классов
    if len(df['TARGET'].unique()) < 2:
      logging.error("Ошибка: недостаточно классов для обучения")
      raise HTTPException(status_code=400, detail="Недостаточно классов для обучения")

    task = Task(item.TaskType, metric=f1_macro)

    automl = TabularNLPAutoML(
        task=task,
        timeout=MODEL_TIMEOUT,
        cpu_limit=os.cpu_count(),
        memory_limit=total_memory,
        reader_params={'n_jobs': os.cpu_count(), 'cv': 5, 'random_state': RANDOM_STATE},
        text_params={'lang': 'ru', 'bert_model': 'DeepPavlov/rubert-base-cased-conversational'},
        general_params={'nested_cv': False, 'use_algos': [['linear_l2', 'lgb', 'cb', 'nn', 'lgb_tuned', 'cb_tuned', 'tabm']]},
    )

  elif item.TaskType == 'reg':

    task = Task(item.TaskType)

    automl = TabularAutoML(
      task=task,
      cpu_limit=os.cpu_count(),
      timeout=MODEL_TIMEOUT,
      memory_limit=total_memory,
      general_params={'nested_cv': False, 'use_algos': [['linear_l2', 'lgb', 'cb', 'nn', 'lgb_tuned', 'cb_tuned', 'tabm']]},
    )

  # Обучение модели
  try:
    await asyncio.to_thread(automl.fit_predict, df, roles=roles, verbose=2)
  except Exception as e:
    logging.error(f"Ошибка при обучении модели: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Ошибка при обучении модели: {str(e)}")

  # Сохраняем модель
  try:
    await asyncio.to_thread(joblib.dump, automl, patchModel)
  except Exception as e:
    logging.error(f"Ошибка при сохранении модели: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Ошибка при сохранении модели: {str(e)}")

  metadata = {
                        'columns': df.columns.tolist(),
                        'dtypes': df.dtypes.to_dict()
                    }

  # Сохраняем метаданные
  try:
    await asyncio.to_thread(joblib.dump, metadata, patchMetadata)
  except Exception as e:
    logging.error(f"Ошибка при сохранении метаданных: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Ошибка при сохранении метаданных: {str(e)}")

  # Добавляем модель в кэш
  app.state.models[NameModel] = automl
  app.state.model_timestamps[NameModel] = datetime.now()
  app.state.model_metadata[NameModel] = metadata

  # Очистка кэша при необходимости
  if len(app.state.models) > MAX_MODEL_CACHE_SIZE:
    # Удаляем самую старую модель
    oldest_model = min(app.state.model_timestamps.items(), key=lambda x: x[1])
    del app.state.models[oldest_model[0]]
    del app.state.model_timestamps[oldest_model[0]]
    logging.info(f"Очищен кэш: удалена модель {oldest_model[0]}")

  logging.info("Обучение завершено")

  return 'Обучение завершено'

@app.post("/predict/", response_model=List[Union[float, int, str, list]])
async def predict(item: Item) -> List[Union[float, int, str, list]]:
  """
  Загружает сохраненную модель и делает предсказание на новых данных
  
  Args:
      item: Данные для предсказания
      
  Returns:
      List[Union[float, int, str]]: Результаты предсказания
      
  Raises:
      HTTPException: При ошибках валидации или предсказания
  """
  logging.info(f"Запуск предсказания модели {item.NameModel}")

  # Валидация входных данных
  if not item.data:
    logging.error("Ошибка: данные не предоставлены")
    raise HTTPException(status_code=400, detail="Данные не предоставлены")
  
  if not item.NameModel:
    logging.error("Ошибка: имя модели не указано")
    raise HTTPException(status_code=400, detail="Имя модели не указано")
  
  if item.TaskType not in ['multiclass', 'binary', 'reg']:
    logging.error(f"Ошибка: неподдерживаемый тип задачи: {item.TaskType}")
    raise HTTPException(status_code=400, detail=f"Неподдерживаемый тип задачи: {item.TaskType}")

  try:
    df = pd.DataFrame(item.data)
  except Exception as e:
    logging.error(f"Ошибка при создании DataFrame: {str(e)}")
    raise HTTPException(status_code=400, detail=f"Ошибка при создании DataFrame: {str(e)}")

  NameModel = item.NameModel
  # Получаем путь к директории модели
  model_dir = './app/' + NameModel
  # Получаем путь к файлу модели
  patchModel = model_dir + '/model.pkl'
  patchMetadata = model_dir + '/metadata.pkl'
  
  # Проверка пути к файлу модели
  model_path = Path(patchModel)
  if not model_path.exists():
    logging.error(f"Файл модели не найден: {patchModel}")
    raise HTTPException(status_code=400, detail=f"Файл модели не найден: {patchModel}")

  # Используем кэширование моделей
  if app.state.models.get(NameModel) is None:
    try:
      # Выполняем синхронную операцию в отдельном потоке
      app.state.models[NameModel] = await asyncio.to_thread(joblib.load, patchModel)
      app.state.model_metadata[NameModel] = await asyncio.to_thread(joblib.load, patchMetadata)
      app.state.model_timestamps[NameModel] = datetime.now()
    except Exception as e:
      logging.error(f"Ошибка при загрузке модели: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Ошибка при загрузке модели: {str(e)}")
  else:
    # Обновляем время последнего использования модели
    if NameModel in app.state.model_timestamps:
        app.state.model_timestamps[NameModel] = datetime.now()

  automl = app.state.models.get(NameModel)
  metadata = app.state.model_metadata.get(NameModel)

  logging.info("Загрузка модели завершена")

  new_df = preprocess_data(metadata, df)

  try:
    # Выполняем синхронную операцию в отдельном потоке
    test_pred = await asyncio.to_thread(automl.predict, new_df)
  except Exception as e:
    logging.error(f"Ошибка при предсказании: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")

  logging.info("Предсказание завершено")

  if item.TaskType == 'multiclass' or item.TaskType == 'binary':
    if hasattr(automl.reader, 'class_mapping') and automl.reader.class_mapping is not None:
        return find_max_indices(test_pred.data.tolist(), automl.reader.class_mapping)
    else:
        # Если class_mapping отсутствует, возвращаем просто индексы
        return [np.argmax(row) for row in test_pred.data.tolist()]
  else:
    return test_pred.data.tolist()

def find_max_indices(arr, mapping):
   """
   Возвращает массив индексов максимальных значений в каждом массиве.
   
   Args:
       arr: Массив предсказаний
       mapping: Словарь сопоставления индексов
       
   Returns:
       List: Список индексов максимальных значений
   """
   max_indices = []
   for sub_arr in arr:
       max_value = max(sub_arr)
       max_index = sub_arr.index(max_value)
       # Исправлено: теперь правильно работаем с mapping
       if mapping is not None and isinstance(mapping, dict) and len(mapping) > 0:
           # Если mapping - словарь, используем его для получения ключа
           # Возвращаем ключ, соответствующий индексу
           keys = list(mapping.keys())
           if max_index < len(keys):
               max_indices.append(keys[max_index])
           else:
               max_indices.append(max_index)
       elif mapping is not None and not isinstance(mapping, dict):
           # Если mapping - не словарь, но не None, возвращаем индекс напрямую
           max_indices.append(max_index)
       else:
           # Если mapping None или пустой, возвращаем индекс напрямую
           max_indices.append(max_index)
   return max_indices

# Предобработка новых данных
def preprocess_data(metadata, new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Предобрабатывает новые данные для соответствия обученной модели
    
    Args:
        metadata: Метаданные обученной модели
        new_data: Новые данные для предсказания
        
    Returns:
        pd.DataFrame: Предобработанные данные
    """
    if not metadata:
        return new_data

    # Создаём DataFrame с нужными колонками и типами
    template_df = pd.DataFrame(columns=metadata['columns'])
    
    # Применяем типы данных к шаблону
    for col, dtype in metadata['dtypes'].items():
        if col in template_df.columns:
            try:
                template_df[col] = template_df[col].astype(dtype)
            except Exception:
                # Если не удалось преобразовать, оставляем как есть
                pass
    
    # Объединяем с новыми данными
    aligned_df = pd.concat([template_df, new_data], ignore_index=True)
    
    # Заполняем пропуски
    fill_values = {}
    for col, dtype in metadata['dtypes'].items():
        if col in aligned_df.columns:
            if np.issubdtype(dtype, np.number):
                fill_values[col] = 0
            elif np.issubdtype(dtype, np.bool_):
                fill_values[col] = False
            else:
                fill_values[col] = 'missing'
    
    aligned_df.fillna(fill_values, inplace=True)
    
    # Возвращаем только нужные колонки в правильном порядке
    return aligned_df[metadata['columns']].reset_index(drop=True)

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
