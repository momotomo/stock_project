@echo off
echo ==========================================
echo 自動売買用: 画面を維持したまま切断します (デバッグ版)
echo ==========================================
echo.

REM セッションIDの抽出
for /f "usebackq" %%i in (`powershell -NoProfile -Command "(Get-Process -Id $PID).SessionId"`) do set CURRENT_SESSION=%%i

echo [DEBUG] 取得したセッションID: %CURRENT_SESSION%
echo セッション [%CURRENT_SESSION%] をコンソールに転送して切断します...
echo.

echo ------------------------------------------
echo ▼ ここからタスク作成のログ ▼
echo ------------------------------------------

schtasks /create /tn "KeepScreenAliveTask" /tr "cmd.exe /c %windir%\System32\tscon.exe %CURRENT_SESSION% /dest:console" /sc once /st 00:00 /ru SYSTEM /rl HIGHEST /f

echo.
echo ------------------------------------------
echo ▼ ここからタスク実行のログ ▼
echo ------------------------------------------

schtasks /run /tn "KeepScreenAliveTask"

echo.
echo ------------------------------------------
echo ▼ ここからタスク削除のログ ▼
echo ------------------------------------------

schtasks /delete /tn "KeepScreenAliveTask" /f

echo.
echo ==========================================
echo ?? 処理が終わりました。この画面に表示されている
echo 「エラー」や「アクセスが拒否されました」などのメッセージを教えてください！
pause