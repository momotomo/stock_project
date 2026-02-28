@echo off
REM =========================================================
REM AI自動取引システム 実行バッチファイル (決定版)
REM =========================================================

REM 1. 作業ディレクトリ（スクリプトがある場所）へ移動
REM /d オプションを付けることで、ドライブを跨ぐ移動を確実に行います
cd /d D:\Users\to-ka\Desktop\AutoTrade

echo [DEBUG] Pythonのフルパスを使って実行を開始します...
echo [DEBUG] 実行中のスクリプト: run_trading_system.py

REM 2. Pythonを実行
REM ※ 以前のエラーログに基づき、Python 3.14のフルパスを直接指定しています
"D:\Users\to-ka\AppData\Local\Programs\Python\Python314\python.exe" run_trading_system.py

echo.
echo [DEBUG] 処理が終了しました。
pause