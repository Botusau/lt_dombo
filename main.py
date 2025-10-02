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
app = FastAPI(title="LTAutoML API", description="API для обучения и предсказания с использованием LightAutoML")

# Хранилище моделей в памяти
app.state.models = {}
# Хранилище времени последнего использования моделей
app.state.model_timestamps = {}

# Константы
RANDOM_STATE = 45  # fixed random state for various reasons
MAX_MODEL_CACHE_SIZE = 10  # Максимальное количество моделей в кэше
MODEL_TIMEOUT = 14400  # Таймаут обучения в секундах (4 часа)

# Установка случайного состояния
np.random.seed(RANDOM_STATE)

class Item(BaseModel):
  """
  Модель данных для запросов API
  """
  data: Any = Field(..., description="Данные для обучения или предсказания")
  NameModel: str = Field(..., description="Имя модели")
  TaskType: str = Field(..., description="Тип задачи: multiclass, binary или reg")
  df_text: Optional[Any] = Field(None, description="Колонка текстовых данных")
  df_drop: Optional[Any] = Field(None, description="Колонки для удаления")

class PredictionResponse(BaseModel):
  """
  Модель ответа для предсказаний
  """
  predictions: List[Union[float, int, str]] = Field(..., description="Результаты предсказания")

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
    if not Path('./app').mkdir():
      logging.error("Ошибка: не удалось создать директорию /app")
      raise HTTPException(status_code=500, detail="Не удалось создать директорию /app")

  NameModel = item.NameModel
  patchModel = './app/' + NameModel + '.pkl' # Получаем путь к файлу модели
  
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
        general_params={'nested_cv': False, 'use_algos': [['linear_l2', 'lgb', 'cb', 'nn', 'lgb_tuned', 'cb_tuned']]},
    )

  elif item.TaskType == 'reg':

    task = Task(item.TaskType)

    automl = TabularAutoML(
      task=task,
      cpu_limit=os.cpu_count(),
      timeout=MODEL_TIMEOUT,
      memory_limit=total_memory,
      general_params={'nested_cv': False, 'use_algos': [['linear_l2', 'lgb', 'cb', 'nn', 'lgb_tuned', 'cb_tuned']]},
    )

  try:
    # Выполняем синхронную операцию в отдельном потоке
    await asyncio.to_thread(automl.fit_predict, df, roles=roles, verbose=2)
  except Exception as e:
    logging.error(f"Ошибка при обучении модели: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Ошибка при обучении модели: {str(e)}")

  try:
    # Выполняем синхронную операцию в отдельном потоке
    await asyncio.to_thread(joblib.dump, automl, patchModel)
  except Exception as e:
    logging.error(f"Ошибка при сохранении модели: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Ошибка при сохранении модели: {str(e)}")

  # Добавляем модель в кэш
  app.state.models[NameModel] = automl
  app.state.model_timestamps[NameModel] = datetime.now()

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
  patchModel = './app/' + NameModel + '.pkl' # Получаем путь к файлу модели
  
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
      app.state.model_timestamps[NameModel] = datetime.now()
    except Exception as e:
      logging.error(f"Ошибка при загрузке модели: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Ошибка при загрузке модели: {str(e)}")
  else:
    # Обновляем время последнего использования модели
    app.state.model_timestamps[NameModel] = datetime.now()

  automl = app.state.models.get(NameModel)
  
  logging.info("Загрузка модели завершена")

  try:
    # Выполняем синхронную операцию в отдельном потоке
    test_pred = await asyncio.to_thread(automl.predict, df)
  except Exception as e:
    logging.error(f"Ошибка при предсказании: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")

  logging.info("Предсказание завершено")

  if item.TaskType == 'multiclass' or item.TaskType == 'binary':
    return find_max_indices(test_pred.data.tolist(), automl.reader.class_mapping)
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
      max_indices.append(list(mapping.keys())[max_index])
  return max_indices

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
