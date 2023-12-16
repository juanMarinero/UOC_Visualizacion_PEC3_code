#!/bin/bash

## run current script
# sudo chmod +x install.sh # privileges
# ./install.sh # run

dir_venv=$HOME/Downloads/UOC_Visualizacion_PEC3_code_venv
dir_repo=$HOME/Downloads/UOC_Visualizacion_PEC3_code_repository

# clone repository
git clone https://github.com/juanMarinero/UOC_Visualizacion_PEC3_code
mv UOC_Visualizacion_PEC3_code $dir_repo

# create virtual environment
python3 -m venv $dir_venv
source $dir_venv/bin/activate

# install dependencies
cd $dir_repo
pip install -r requirements.txt

# run notebook
jupyter notebook
