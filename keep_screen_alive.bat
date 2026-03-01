@echo off
echo ==========================================
echo 自動売買用: 画面を維持したまま切断します (待機時間追加版)
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

REM 裏側のエラーを見えるように %PUBLIC%\tscon_error.txt に出力します
schtasks /create /tn "KeepScreenAliveTask" /tr "cmd.exe /c %windir%\System32\tscon.exe %CURRENT_SESSION% /dest:console > %PUBLIC%\tscon_error.txt 2>&1" /sc once /st 00:00 /ru SYSTEM /rl HIGHEST /f

echo.
echo ------------------------------------------
echo ▼ ここからタスク実行のログ ▼
echo ------------------------------------------

schtasks /run /tn "KeepScreenAliveTask"

echo.
echo ? タスクが裏側で実行されるのを3秒間待機しています...
timeout /t 3 /nobreak >nul

echo.
echo ------------------------------------------
echo ▼ ここからタスク削除のログ ▼
echo ------------------------------------------

schtasks /delete /tn "KeepScreenAliveTask" /f

echo.
echo ==========================================
echo ?? これでも画面が閉じない場合、Windowsからの隠しエラーが出ています。
echo C:\Users\Public フォルダの中にある「tscon_error.txt」を開いて、
echo 書かれている内容を教えてください！
pause