@echo off
echo ==========================================
echo 自動売買用: 画面を維持したまま切断します
echo ==========================================
echo.
echo ※この画面が消えると同時にWorkSpacesのウィンドウが閉じますが、
echo 　裏側では画面が「開いたまま（ロックされない状態）」で維持されます。
echo.

REM 現在のユーザーのセッションIDを取得し、コンソールセッションに強制転送して切断する裏技
for /f "skip=1 tokens=3" %%s in ('query user %USERNAME%') do (
  %windir%\System32\tscon.exe %%s /dest:console
)