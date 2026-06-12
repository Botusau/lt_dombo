#!/bin/bash

pip install --upgrade pip
pip install -U torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install -U -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000