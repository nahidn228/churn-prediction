@echo off
cd /d "%~dp0"
echo ================================================================
echo   Launching Customer Segmentation & Churn Intelligence Engine
echo ================================================================
if exist "C:\Users\BLG\AppData\Local\Python\bin\python.exe" (
    "C:\Users\BLG\AppData\Local\Python\bin\python.exe" -m streamlit run app.py
) else if exist "C:\Users\BLG\AppData\Local\Python\pythoncore-3.14-64\Scripts\streamlit.exe" (
    "C:\Users\BLG\AppData\Local\Python\pythoncore-3.14-64\Scripts\streamlit.exe" run app.py
) else (
    python -m streamlit run app.py
)
pause
