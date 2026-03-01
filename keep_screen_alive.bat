@echo off
echo ==========================================
echo 自動売買用: 画面を維持したまま切断します
echo ==========================================
echo.

REM PowerShell経由で現在のプロセスのセッションIDを確実に抽出（列ズレ対策）
for /f "usebackq" %%i in (`powershell -NoProfile -Command "(Get-Process -Id $PID).SessionId"`) do set CURRENT_SESSION=%%i

echo [DEBUG] 取得したセッションID: %CURRENT_SESSION%
echo セッション [%CURRENT_SESSION%] をコンソールに転送して切断します...
echo.

REM 抽出したセッションIDを使って直接転送
%windir%\System32\tscon.exe %CURRENT_SESSION% /dest:console

echo.
echo ?? もしこの画面が閉じずに残っている場合、エラーが発生しています。
echo おそらく「管理者として実行」されていないことが原因です。
pause