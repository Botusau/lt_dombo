#!/bin/bash

pip install --upgrade pip
#pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000