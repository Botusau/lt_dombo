#!/bin/bash

if [[ -d ./lt_dombo ]]; then
    echo "Removing Lock"
    rm -rf ./lt_dombo
fi

git clone https://github.com/Botusau/lt_dombo.git

cd ./lt_dombo/

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r ./lt_dombo/requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000