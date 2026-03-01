@echo off
echo ==========================================
echo 自動売買用: 画面を維持したまま切断します
echo ==========================================
echo.

REM PowerShell経由で現在のプロセスのセッションIDを抽出
for /f "usebackq" %%i in (`powershell -NoProfile -Command "(Get-Process -Id $PID).SessionId"`) do set CURRENT_SESSION=%%i

echo [DEBUG] 取得したセッションID: %CURRENT_SESSION%
echo セッション [%CURRENT_SESSION%] をコンソールに転送して切断します...
echo.

REM ========================================================
REM ★ エラー 7045 (アクセス拒否) 対策 ★
REM SYSTEM権限でtsconを実行するため、一時的なタスクを作成して
REM 裏側で即実行し、すぐに削除するハックを使用します。
REM ========================================================

echo 権限昇格の準備中...
schtasks /create /tn "KeepScreenAliveTask" /tr "%windir%\System32\tscon.exe %CURRENT_SESSION% /dest:console" /sc once /st 00:00 /ru SYSTEM /rl HIGHEST /f >nul 2>&1

echo 切断処理を実行します...
schtasks /run /tn "KeepScreenAliveTask" >nul 2>&1

REM 後片付け（この行は切断と同時に走るか、無視されます）
schtasks /delete /tn "KeepScreenAliveTask" /f >nul 2>&1

echo.
echo ?? もしこの画面が閉じずに残っている場合、エラーが発生しています。
echo ショートカットが「管理者として実行」に設定されているか確認してください。
pause