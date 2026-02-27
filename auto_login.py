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
# kabuステーション 完全自動ログインスクリプト (最新UI＆メール対応版)
# ---------------------------------------------------------
# 【動作要件】
# 1. pip install pyautogui pyperclip psutil opencv-python
# 2. 画面のスケーリング（拡大率）は 100% を推奨
# =========================================================

# --- 基本設定 ---
KABU_APP_PATH = r"D:\Users\to-ka\AppData\Local\kabuStation\KabuS.exe"
LOGIN_PASSWORD = "Tr7smv_jnxg"  # 証券口座のパスワード

# --- メール（IMAP）設定 ---
IMAP_SERVER = "imap.gmail.com"
EMAIL_ADDRESS = "tomo.19851206@gmail.com"
EMAIL_PASSWORD = "ehxqpbtcpdrtotsy" # Gmailのアプリパスワード(16桁)

def kill_existing_process():
    """既存のkabuステーションを終了"""
    print("🔄 既存プロセスの確認中...")
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == 'kabu.station.exe':
            proc.kill()
            time.sleep(3)

def get_otp_from_email(max_retries=12, wait_seconds=10):
    """メールからワンタイム認証コードを抽出"""
    print("📧 メールボックスをスキャン中...")
    for attempt in range(max_retries):
        try:
            mail = imaplib.IMAP4_SSL(IMAP_SERVER)
            mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            mail.select("inbox")
            
            # 最近のメールを取得
            status, messages = mail.search(None, "ALL")
            mail_ids = messages[0].split()
            
            if mail_ids:
                # 最新のメール5件をチェック
                for i in reversed(mail_ids[-5:]):
                    res, msg_data = mail.fetch(i, "(RFC822)")
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            
                            # 件名のデコード
                            subject, encoding = decode_header(msg["Subject"])[0]
                            if isinstance(subject, bytes):
                                subject = subject.decode(encoding if encoding else "utf-8", errors="ignore")
                            
                            # スクショに基づく条件: 件名に「ワンタイム」または「認証コード」を含むか
                            if "ワンタイム" in subject or "認証コード" in subject:
                                body = ""
                                
                                # 本文のデコード（日本のメール特有の文字コードに対応）
                                if msg.is_multipart():
                                    for part in msg.walk():
                                        if part.get_content_type() == "text/plain":
                                            charset = part.get_content_charset() or 'utf-8'
                                            body = part.get_payload(decode=True).decode(charset, errors="ignore")
                                            break
                                else:
                                    charset = msg.get_content_charset() or 'utf-8'
                                    body = msg.get_payload(decode=True).decode(charset, errors="ignore")
                                
                                # スクショに基づく抽出ロジック
                                # 「■ワンタイム認証コード」の直後にある6桁の数字を狙い撃ち
                                match = re.search(r'■ワンタイム認証コード\s*(\d{6})', body)
                                
                                # もし上の条件で見つからなければ、単独の6桁の数字を探す（保険）
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
    """画像に基づいたログインシーケンス"""
    print("🚀 アプリを起動します...")
    subprocess.Popen([KABU_APP_PATH])
    time.sleep(25) # 起動待機
    
    # 画面をアクティブにするため中央をクリック
    w, h = pyautogui.size()
    pyautogui.click(w/2, h/2)
    time.sleep(1)

    # --- Step 1: ログイン実行 (パスワード保存済み) ---
    print("⌨️ ログインボタンを押下します (ID/PW保存済み)...")
    
    # パスワード入力欄にフォーカスがあり、値も入っているため、Enterキーだけで送信可能
    pyautogui.press('enter')
    print("✅ ログイン要求の送信完了")
    time.sleep(10) # OTP画面への遷移待機

    # --- Step 2: OTP入力 ---
    otp = get_otp_from_email()
    if not otp:
        print("❌ 認証コードの取得に失敗しました。")
        return

    print(f"🔑 画面に認証コード ({otp}) を入力中...")
    pyperclip.copy(otp)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(1)
    
    # 「続ける」ボタンを押す（Enterキー）
    pyautogui.press('enter')
    
    print("🎊 ログインシーケンス完了。アプリの起動を待ちます。")
    time.sleep(30)

if __name__ == "__main__":
    kill_existing_process()
    perform_login()
    pyperclip.copy("") # セキュリティのためクリップボード消去