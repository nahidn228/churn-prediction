Set-Location -LiteralPath $PSScriptRoot
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Launching Customer Segmentation & Churn Intelligence Engine   " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

if (Test-Path "C:\Users\BLG\AppData\Local\Python\bin\python.exe") {
    & "C:\Users\BLG\AppData\Local\Python\bin\python.exe" -m streamlit run app.py
} elseif (Test-Path "C:\Users\BLG\AppData\Local\Python\pythoncore-3.14-64\Scripts\streamlit.exe") {
    & "C:\Users\BLG\AppData\Local\Python\pythoncore-3.14-64\Scripts\streamlit.exe" run app.py
} elseif (Get-Command streamlit -ErrorAction SilentlyContinue) {
    streamlit run app.py
} else {
    python -m streamlit run app.py
}
