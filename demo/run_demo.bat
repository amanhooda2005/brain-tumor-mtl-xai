@echo off
REM Start the live demo (Windows). Extra options are passed through.
cd /d "%~dp0\.."
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python demo\app.py %*
pause
