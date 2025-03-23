import os
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from lightautoml.tasks import Task
import numpy as np
from lightautoml.automl.presets.text_presets import TabularNLPAutoML
import joblib
from sklearn.metrics import f1_score

app = FastAPI()
app.state.models = {}

RANDOM_STATE = 45  # fixed random state for various reasons

class Item(BaseModel):
  data: Any

def f1_macro(y_true, y_pred):
  return f1_score(y_true, np.argmax(y_pred, axis=1), average='macro')

@app.get("/")
async def home_page():
  """
  Возвращает приветственное сообщение.
  """
  return {"message": "Привет, LT!"}


@app.post("/fit_predict/")
async def fit_predict(item: Item):
  """
  Обучает модель на данных, сохраняет ее и возвращает результаты обучения.
  """
  df = pd.DataFrame(item.data)

  value_counts = df['TARGET'].value_counts() # Подсчитываем количество повторений каждого значения в колонке TARGET
  values_to_drop = value_counts[value_counts == 1].index # Находим значения, которые встречаются только один раз
  df = df[~df['TARGET'].isin(values_to_drop)] # Удаляем строки, где значение в колонке TARGET встречается только один раз

  df = clean_df(df)

  df.to_csv('sdata.csv', index=False)

  task = Task('multiclass', metric=f1_macro)
  roles = {
      'target': 'TARGET',
      'text': ['Materials'],
      'drop': ['Key'],
  }

  np.random.seed(RANDOM_STATE)

  automl = TabularNLPAutoML(
      task=task,
      timeout = 14400,
      reader_params = {'n_jobs': os.cpu_count(), 'cv': 5, 'random_state': RANDOM_STATE},
      text_params = {'lang': 'ru', 'bert_model': 'DeepPavlov/rubert-base-cased-conversational'},
      general_params={'nested_cv': False, 'use_algos': [['linear_l2', 'lgb', 'cb', 'nn']]},    
  )

  automl.fit_predict(df, roles=roles, verbose=2)

  joblib.dump(automl, "/app/model.pkl")

  app.state.models['model'] = automl

  return 'Обучение завершено'


@app.post("/predict/")
async def predict(item: Item):
  """
  Загружает сохраненную модель, делает предсказание на новых данных и возвращает результаты.
  """
  df = pd.DataFrame(item.data)

  df = clean_df(df)

  df.to_csv('items.csv', index=False)

  if app.state.models.get("model") is None:
    app.state.models['model'] = joblib.load("/app/model.pkl")

  automl = app.state.models.get("model")
  
  test_pred = automl.predict(df)

  return find_max_indices(test_pred.data.tolist(), automl.reader.class_mapping)


def find_max_indices(arr, mapping):
  """
  Возвращает массив индексов максимальных значений в каждом массиве.
  """
  max_indices = []
  for sub_arr in arr:
      max_value = max(sub_arr)
      max_index = sub_arr.index(max_value)
      max_indices.append(list(mapping.keys())[max_index])
  return max_indices

def clean_df(df):
  """
  Очищает текст от лишних символов.
  """
  df['Materials'] = df['Materials'].str.lower()
  df['Materials'] = df['Materials'].replace("[0-9!#()$\,\'\-\.*+/:;<=>?@[\]^_`{|}\"]+", ' ', regex=True)
  df['Materials'] = df['Materials'].replace(r'\s+', ' ', regex=True)

  return df