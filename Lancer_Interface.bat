@echo off
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat goto run
set PYCMD=
py -3.12 --version >nul 2>&1 && set PYCMD=py -3.12
if not defined PYCMD python --version 2>&1 | findstr /b /c:"Python 3.12" >nul && set PYCMD=python
if not defined PYCMD goto nopython
%PYCMD% -m venv .venv
if errorlevel 1 goto fail
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
python -m pip install -r requirements.txt -r requirements_app.txt
if errorlevel 1 goto fail
:run
call .venv\Scripts\activate.bat
python -m streamlit run app.py
pause
exit /b 0
:nopython
echo Python 3.12 est introuvable. Installe-le depuis python.org en cochant Add python.exe to PATH,
echo ou dans un terminal : winget install Python.Python.3.12 ... puis rouvre ce fichier.
pause
exit /b 1
:fail
echo L'installation a echoue. Supprime le dossier .venv puis relance, ou copie le message d'erreur ci-dessus.
pause
exit /b 1