@echo off
REM EcoSegrega - instalación solo CPU
python -m pip install --upgrade pip
pip install virtualenv
virtualenv venv
call venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requisitos.txt
pip install -r requisitos-extra.txt
python etapas\e0_entorno.py
pip freeze > requisitos-congelados.txt
pause
