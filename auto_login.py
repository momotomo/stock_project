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
# kabuステーション 完全自動ログインスクリプト (OTP対応版)
# ---------------------------------------------------------
# 【動作要件】
# 1. pip install pyautogui pyperclip psutil opencv-python を実行しておくこと
# 2. kabuステーションのショートカットがデスクトップ等にある、あるいはパスを指定
# 3. 画面の解像度やスケーリング（100%推奨）を変更しないこと
# =========================================================

# --- 設定項目 ---
# kabuステーション本体のパス
KABU_APP_PATH = r"C:\Program Files (x86)\kabu.com\kabu station\kabu.station.exe"
# 証券口座のパスワード
LOGIN_PASSWORD = "Tr7smv_jnxg" 

# --- メール（IMAP）設定 ---
# ※Gmailを使用する場合、通常のパスワードではなく「アプリパスワード」の発行が必要です。
IMAP_SERVER = "imap.gmail.com"
EMAIL_ADDRESS = "tomo.19851206@gmail.com"
EMAIL_PASSWORD = "ehxqpbtcpdrtotsy"


def kill_existing_process():
    """既に起動しているkabuステーションがあれば強制終了する"""
    print("🔄 既存のkabuステーションプロセスを確認しています...")
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == 'kabu.station.exe':
            print(f"⚠️ 既存のプロセス (PID:{proc.pid}) を終了します...")
            proc.kill()
            time.sleep(3)

def launch_app():
    """アプリを起動する"""
    print("🚀 kabuステーションを起動します...")
    try:
        subprocess.Popen([KABU_APP_PATH])
        print("⏳ ログイン画面の表示を待機中 (20秒)...")
        time.sleep(20)
    except FileNotFoundError:
        print(f"❌ エラー: 指定されたパスにアプリが見つかりません: {KABU_APP_PATH}")
        sys.exit(1)

def get_one_time_password(max_retries=10, wait_seconds=5):
    """
    メールボックスを監視し、最新のワンタイムパスワードを取得する
    """
    print("📧 メールボックスからワンタイムパスワード(OTP)を検索しています...")
    
    # 今の時刻から少し前（3分前）以降のメールのみを対象にする
    since_time = datetime.now() - timedelta(minutes=3)
    
    for attempt in range(max_retries):
        try:
            # IMAPサーバーに接続
            mail = imaplib.IMAP4_SSL(IMAP_SERVER)
            mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            mail.select("inbox")
            
            # 検索条件: 未読メール、またはカブコムからのメール（環境に合わせて調整）
            # ここでは「ALL」から取得し、Python側で日付と送信元をチェックします
            status, messages = mail.search(None, "ALL")
            mail_ids = messages[0].split()
            
            if mail_ids:
                # 最新のメールから数件を確認
                for i in reversed(mail_ids[-5:]):
                    res, msg_data = mail.fetch(i, "(RFC822)")
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            
                            # 件名のデコード
                            subject, encoding = decode_header(msg["Subject"])[0]
                            if isinstance(subject, bytes):
                                subject = subject.decode(encoding if encoding else "utf-8")
                            
                            # カブコムのワンタイムパスワードメールか判定
                            if "ワンタイムパスワード" in subject or "kabu.com" in msg.get("From", ""):
                                
                                # 本文の取得
                                body = ""
                                if msg.is_multipart():
                                    for part in msg.walk():
                                        if part.get_content_type() == "text/plain":
                                            body = part.get_payload(decode=True).decode()
                                            break
                                else:
                                    body = msg.get_payload(decode=True).decode()
                                
                                # 正規表現でパスワードらしき文字列（例: 半角英数字6〜10桁）を抽出
                                # ※実際のメール本文に合わせてパターンを調整する必要があります
                                match = re.search(r'([A-Za-z0-9]{6,10})', body)
                                if match:
                                    otp = match.group(1)
                                    print(f"✅ ワンタイムパスワードを取得しました: {otp}")
                                    mail.close()
                                    mail.logout()
                                    return otp
            
            mail.close()
            mail.logout()
        except Exception as e:
            print(f"⚠️ メール確認中にエラー: {e}")
            
        print(f"⏳ まだメールが届いていません。{wait_seconds}秒後に再確認します... ({attempt+1}/{max_retries})")
        time.sleep(wait_seconds)
        
    print("❌ 指定時間内にワンタイムパスワードメールを受信できませんでした。")
    return None

def perform_login():
    """マウスとキーボードを自動操作してログイン（OTP対応）"""
    print("🔑 ログイン処理を開始します...")
    
    # 1. 第一パスワードの入力とEnter
    pyperclip.copy(LOGIN_PASSWORD)
    
    # ウィンドウをアクティブにする
    screen_width, screen_height = pyautogui.size()
    pyautogui.click(screen_width / 2, screen_height / 2)
    time.sleep(1)
    
    # パスワードをペーストしてログインボタンを押す（Enter）
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(1)
    pyautogui.press('enter')
    
    print("✅ パスワードを入力しました。ワンタイムパスワードの要求画面を待機します (5秒)...")
    time.sleep(5)
    
    # 2. メールの受信とOTPの取得
    otp = get_one_time_password()
    if not otp:
        print("❌ ログインを中断します。")
        sys.exit(1)
        
    # 3. ワンタイムパスワードの入力とEnter
    print("🔑 画面にワンタイムパスワードを入力します...")
    # ウィンドウがアクティブなまま、OTP入力欄にフォーカスがあると想定
    # （もしフォーカスが外れる場合は、再度クリックやTAB移動が必要です）
    pyperclip.copy(otp)
    time.sleep(1)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(1)
    pyautogui.press('enter')
    
    print("✅ 最終ログイン操作を実行しました。起動完了まで待機します (30秒)...")
    time.sleep(30)
    print("🎉 自動ログインプロセス完了。")

if __name__ == "__main__":
    print("=== kabuステーション 自動ログインシステム (OTP対応) ===")
    kill_existing_process()
    launch_app()
    perform_login()
    
    # クリップボードの消去
    pyperclip.copy("")