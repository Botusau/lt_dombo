#!/bin/bash

pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -U -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000