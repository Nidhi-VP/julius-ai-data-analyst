@echo off
title Julius AI Data Analyst
cd /d "%~dp0"
echo Starting Julius AI Data Analyst...
start http://localhost:8501
python -m streamlit run app.py
pause
