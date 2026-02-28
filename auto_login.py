import time
import subprocess
import pyautogui
import pyperclip
import psutil
import sys
import imaplib
import email
from email.header import decode_header
import re
from datetime import datetime, timedelta

# =========================================================
# kabuステーション 完全自動ログインスクリプト (パス修正版)
# =========================================================

# --- 基本設定 ---
# 💡 プロセス名に合わせて実行ファイル名も「KabuS.exe」に変更しました
KABU_APP_PATH = r"C:\Program Files (x86)\kabu.com\kabu station\KabuS.exe"
TARGET_PROCESS = "KabuS.exe"

# --- メール（IMAP）設定 ---
IMAP_SERVER = "imap.gmail.com"
EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password" 

def kill_existing_process():
    """既存のkabuステーションを終了"""
    print(f"🔄 既存プロセス ({TARGET_PROCESS}) の確認中...")
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'].lower() == TARGET_PROCESS.lower():
                proc.kill()
                time.sleep(3)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def get_otp_from_email(max_retries=12, wait_seconds=10):
    """メールからワンタイム認証コードを抽出"""
    print("📧 メールボックスをスキャン中...")
    for attempt in range(max_retries):
        try:
            mail = imaplib.IMAP4_SSL(IMAP_SERVER)
            mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            mail.select("inbox")
            status, messages = mail.search(None, "ALL")
            mail_ids = messages[0].split()
            
            if mail_ids:
                for i in reversed(mail_ids[-5:]):
                    res, msg_data = mail.fetch(i, "(RFC822)")
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            subject, encoding = decode_header(msg["Subject"])[0]
                            if isinstance(subject, bytes):
                                subject = subject.decode(encoding if encoding else "utf-8", errors="ignore")
                            
                            if "ワンタイム" in subject or "認証コード" in subject:
                                body = ""
                                if msg.is_multipart():
                                    for part in msg.walk():
                                        if part.get_content_type() == "text/plain":
                                            charset = part.get_content_charset() or 'utf-8'
                                            body = part.get_payload(decode=True).decode(charset, errors="ignore")
                                            break
                                else:
                                    charset = msg.get_content_charset() or 'utf-8'
                                    body = msg.get_payload(decode=True).decode(charset, errors="ignore")
                                
                                match = re.search(r'■ワンタイム認証コード\s*(\d{6})', body)
                                if not match:
                                    match = re.search(r'(?<!\d)(\d{6})(?!\d)', body)
                                    
                                if match:
                                    otp = match.group(1)
                                    print(f"✅ メールの解析成功！認証コードを発見: {otp}")
                                    mail.logout()
                                    return otp
            mail.logout()
        except Exception as e:
            print(f"⚠️ メール取得・解析エラー: {e}")
            
        print(f"⏳ メール待機中... ({attempt+1}/{max_retries})")
        time.sleep(wait_seconds)
    return None

def perform_login():
    """画像認識に基づいたログインシーケンス"""
    print("🚀 アプリを起動します...")
    # 💡 ここで FileNotFoundError が出たのは KABU_APP_PATH の場所が間違っていたためです
    try:
        subprocess.Popen([KABU_APP_PATH])
    except FileNotFoundError:
        print(f"❌ エラー: {KABU_APP_PATH} が見つかりません。")
        print("デスクトップのショートカットを右クリックして『プロパティ』から正しいパスを確認してください。")
        sys.exit(1)
    
    print("⏳ ログイン画面の表示を待機しています（最大60秒）...")
    login_btn_pos = None
    for i in range(60):
        try:
            login_btn_pos = pyautogui.locateCenterOnScreen('login_btn.png', confidence=0.8)
            if login_btn_pos:
                break
        except Exception:
            pass
        if i % 10 == 0 and i > 0:
            print(f"...待機中 ({i}秒経過)")
        time.sleep(1)

    if not login_btn_pos:
        print("❌ 画面上に「login_btn.png」が見つかりませんでした。")
        return

    print(f"🎯 ログインボタンを発見しました。クリックします...")
    pyautogui.click(login_btn_pos)
    time.sleep(10)

    otp = get_otp_from_email()
    if not otp:
        print("❌ 認証コードの取得に失敗しました。")
        return

    print(f"🔑 画面に認証コード ({otp}) を入力中...")
    pyperclip.copy(otp)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(1)
    pyautogui.press('enter')
    
    print("🎊 ログインシーケンス完了。アプリが完全に起動するまで60秒間待機します。")
    for i in range(6):
        time.sleep(10)
        print(f"...起動待機中 ({ (i+1)*10 }秒経過)")
    
    print("✅ すべての準備が整いました。")

if __name__ == "__main__":
    kill_existing_process()
    perform_login()
    pyperclip.copy("")