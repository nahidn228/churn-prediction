#!/bin/bash
echo "================================================================"
echo "  Launching Customer Segmentation & Churn Intelligence Engine   "
echo "================================================================"
cd "$(dirname "$0")" || exit 1
export PYTHONPATH=".:$PYTHONPATH"

if [ -f "/c/Users/BLG/AppData/Local/Python/bin/python.exe" ]; then
    /c/Users/BLG/AppData/Local/Python/bin/python.exe -m streamlit run app.py
elif [ -f "/c/Users/BLG/AppData/Local/Python/pythoncore-3.14-64/Scripts/streamlit.exe" ]; then
    /c/Users/BLG/AppData/Local/Python/pythoncore-3.14-64/Scripts/streamlit.exe run app.py
elif command -v streamlit >/dev/null 2>&1; then
    streamlit run app.py
elif command -v python >/dev/null 2>&1; then
    python -m streamlit run app.py
else
    echo "ERROR: Streamlit executable not found. Please verify Python installation."
fi
