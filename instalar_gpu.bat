@echo off
REM EcoSegrega - instalación con GPU NVIDIA (controlador que soporte CUDA 12.6 o más)
python -m pip install --upgrade pip
pip install virtualenv
virtualenv venv
call venv\Scripts\activate
REM No usar "pip install torch" a secas: desde PyTorch 2.11 trae CUDA 13 (pide controlador 580+).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requisitos.txt
pip install -r requisitos-extra.txt
python etapas\e0_entorno.py
pip freeze > requisitos-congelados.txt
echo Listo. En cada consola nueva:  venv\Scripts\activate
pause
