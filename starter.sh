#!/bin/bash

cd /app
mkdir lt_dombo
rm -rf lt_dombo

git clone https://github.com/Botusau/lt_dombo.git

cd ./lt_dombo

sh run.sh