@echo off
title AL-QNN Quantum Studio
echo ======================================================================
echo   KHOI DONG AL-QNN QUANTUM STUDIO (UI/UX DASHBOARD)
echo   Dia chi: http://localhost:8501
echo ======================================================================
cd /d "%~dp0al_qnn_studio"
start http://localhost:8501
python -m http.server 8501
pause
