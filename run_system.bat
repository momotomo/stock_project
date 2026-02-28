@echo off
echo [DEBUG] Starting AI Trading System...

REM 1. スクリプトのあるフォルダへ移動
cd /d "D:\Users\to-ka\Desktop\AutoTrade"

echo [DEBUG] Working Directory: %cd%

REM 2. Pythonを実行（パスの末尾まで正確に指定）
"D:\Users\to-ka\AppData\Local\Programs\Python\Python314\python.exe" run_trading_system.py

echo.
echo [DEBUG] Script ended.
pause