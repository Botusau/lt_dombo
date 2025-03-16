#!/bin/bash

if [[ -d ./lt_dombo ]]; then
    rm -rf ./lt_dombo
fi

git clone https://github.com/Botusau/lt_dombo.git

./app/lt_dombo/run.sh