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
# kabuステーション 完全自動ログインスクリプト (無条件待機強化版)
# =========================================================

# --- 基本設定 ---
KABU_APP_PATH = r"D:\Users\to-ka\AppData\Local\kabuStation\KabuS.exe"
TARGET_PROCESS = "KabuS.exe"

# --- メール（IMAP）設定 ---
IMAP_SERVER = "imap.gmail.com"
EMAIL_ADDRESS = "tomo.19851206@gmail.com"
EMAIL_PASSWORD = "ehxqpbtcpdrtotsy" 

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
    try:
        subprocess.Popen([KABU_APP_PATH])
    except FileNotFoundError:
        print(f"❌ エラー: {KABU_APP_PATH} が見つかりません。")
        sys.exit(1)
    
    # 💡 【今回の変更】見つけ次第動くのではなく、まずは無条件で30秒待機
    FIXED_WAIT = 30 
    print(f"⏳ アプリの安定化のため、無条件で {FIXED_WAIT} 秒間待機します...")
    for i in range(FIXED_WAIT):
        time.sleep(1)
        if (i + 1) % 10 == 0:
            print(f"...待機中 ({i + 1}/{FIXED_WAIT}秒経過)")

    print("🔎 ログインボタンの検索を開始します...")
    login_btn_pos = None
    # 既に60秒待っているので、ここは短めの10秒間のリトライに設定
    for _ in range(10):
        try:
            login_btn_pos = pyautogui.locateCenterOnScreen('login_btn.png', confidence=0.8)
            if login_btn_pos:
                break
        except Exception:
            pass
        time.sleep(1)

    if not login_btn_pos:
        print("❌ 待機しましたが「login_btn.png」が見つかりませんでした。")
        return

    print(f"🎯 ログインボタンをクリックします...")
    pyautogui.click(login_btn_pos)
    time.sleep(10)

    otp = get_otp_from_email()
    if not otp:
        print("❌ 認証コードの取得に失敗しました。")
        return

    print(f"🔑 画面に認証コード ({otp}) を入力中...")
    pyperclip.copy(otp)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(2) # 入力後の微調整待機
    pyautogui.press('enter')
    
    # ログイン後のメイン画面起動待ち（ここも無条件30秒）
    print(f"🎊 ログイン完了。メイン画面の起動をさらに {FIXED_WAIT} 秒間待ちます。")
    for i in range(6):
        time.sleep(10)
        print(f"...最終起動待機中 ({ (i+1)*10 }/{FIXED_WAIT}秒経過)")
    
    print("✅ すべての準備が整いました。")

if __name__ == "__main__":
    kill_existing_process()
    perform_login()
    pyperclip.copy("")