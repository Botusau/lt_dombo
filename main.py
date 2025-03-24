import os
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from lightautoml.tasks import Task
import numpy as np
from lightautoml.automl.presets.text_presets import TabularNLPAutoML
from lightautoml.automl.presets.tabular_presets import TabularAutoML
import joblib
from sklearn.metrics import f1_score

app = FastAPI()
app.state.models = {}

RANDOM_STATE = 45  # fixed random state for various reasons
np.random.seed(RANDOM_STATE)

class Item(BaseModel):
  data: Any
  NameModel: str
  TaskType: str
  df_text: Any
  df_drop: Any


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
  df.to_csv('sdata.csv', index=False)

  NameModel = item.NameModel
  patchModel = '/app/' + NameModel + '.pkl' # Получаем путь к файлу модели

  roles = {
        'target': 'TARGET',
    }
  
  if item.df_text != None:
    roles['text'] = item.df_text

  if item.df_drop != None:
    roles['drop'] = item.df_drop

  if item.TaskType == 'multiclass' or item.TaskType == 'binary':

    value_counts = df['TARGET'].value_counts() # Подсчитываем количество повторений каждого значения в колонке TARGET
    values_to_drop = value_counts[value_counts == 1].index # Находим значения, которые встречаются только один раз
    df = df[~df['TARGET'].isin(values_to_drop)] # Удаляем строки, где значение в колонке TARGET встречается только один раз

    task = Task(item.TaskType, metric=f1_macro)

    automl = TabularNLPAutoML(
        task=task,
        timeout = 14400,
        memory_limit=6, 
        reader_params = {'n_jobs': os.cpu_count(), 'cv': 5, 'random_state': RANDOM_STATE},
        text_params = {'lang': 'ru', 'bert_model': 'DeepPavlov/rubert-base-cased-conversational'},
        general_params={'nested_cv': False, 'use_algos': [['linear_l2', 'lgb', 'cb', 'nn', 'lgb_tuned', 'cb_tuned']]},    
    )

  elif item.TaskType == 'reg':

    task = Task(item.TaskType)

    automl = TabularAutoML(
      task=task,
      timeout = 14400,
      memory_limit=6,
      general_params={'nested_cv': False, 'use_algos': [['linear_l2', 'lgb', 'cb', 'nn', 'lgb_tuned', 'cb_tuned']]},    
    )

  automl.fit_predict(df, roles=roles, verbose=2)

  joblib.dump(automl, patchModel)

  app.state.models[NameModel] = automl

  return 'Обучение завершено'


@app.post("/predict/")
async def predict(item: Item):
  """
  Загружает сохраненную модель, делает предсказание на новых данных и возвращает результаты.
  """
  df = pd.DataFrame(item.data)
  NameModel = item.NameModel
  patchModel = '/app/' + NameModel + '.pkl' # Получаем путь к файлу модели

  df.to_csv('items.csv', index=False)

  if app.state.models.get(NameModel) is None:
    app.state.models[NameModel] = joblib.load(patchModel)

  automl = app.state.models.get(NameModel)
  
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