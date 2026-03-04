import sys
import subprocess
import time
import logging
import psutil
import yaml
import os

# =========================================================
# 自動取引システム 統合実行ランナー (Kaggle API直結版)
# =========================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = "settings.yml"

# ⚠️ 【重要】ここにあなたのKaggleノートブックのURLの一部（ID）を入力してください
# 例： https://www.kaggle.com/code/taro/stock-ai だったら "taro/stock-ai" と書く
KAGGLE_NOTEBOOK_SLUG = "momotomo/stock-ai-trainer"

def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"❌ 設定ファイルの読み込みエラー: {e}")
        return None

def kill_kabu_station():
    """kabuステーションのプロセス(KabuS.exe)を確実に終了させる"""
    logger.info("🧹 kabuステーションのプロセスを確認・終了します...")
    target_name = "KabuS.exe"
    killed_count = 0
    
    for proc in psutil.process_iter(['name']):
        try:
            p_name = (proc.info.get('name') or "").lower()
            if p_name == target_name.lower():
                logger.info(f"🎯 起動中のプロセスを発見 (PID: {proc.pid})。終了します...")
                proc.kill()
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
            
    if killed_count > 0:
        logger.info(f"✅ {killed_count}個のプロセスを終了しました。")
        time.sleep(3)
    else:
        logger.info(f"ℹ️ 起動中の {target_name} は見つかりませんでした。")

# 🔥 変更：Git Pullではなく、Kaggle APIから直接モデルをダウンロードする関数
def download_models_from_kaggle():
    """Kaggleから最新のAIモデルを直接ダウンロードする"""
    logger.info(f"🌐 Kaggleから最新のAIモデルをダウンロードします ({KAGGLE_NOTEBOOK_SLUG})...")
    
    # ダウンロード先の models フォルダを作成（なければ）
    os.makedirs("models", exist_ok=True)
    
    try:
        # Kaggle APIを使ってOutputファイルを models フォルダにダウンロード (※ --unzip を削除)
        result = subprocess.run(
            ["kaggle", "kernels", "output", KAGGLE_NOTEBOOK_SLUG, "-p", "models"],
            capture_output=True, text=True, check=True
        )
        logger.info(f"✅ Kaggleモデルのダウンロード完了:\n{result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Kaggleからのダウンロードエラー:\n{e.stderr.strip()}")
        logger.error("👉 ヒント: コマンドプロンプトで 'pip install kaggle' が実行されているか、'~/.kaggle/kaggle.json' が正しく配置されているか確認してください。")
    except FileNotFoundError:
        logger.error("❌ 'kaggle' コマンドが見つかりません。'pip install kaggle' を実行してください。")

def main():
    logger.info("=========================================")
    logger.info("🌅 自動取引システム 統合ランナー起動")
    logger.info("=========================================")
    
    config = load_config()
    is_production = config.get("IS_PRODUCTION", False) if config else False

    if is_production:
        logger.info("🔴 【警告】本番モード(IS_PRODUCTION=True)で起動しています。実際の資金が動きます。")
    else:
        logger.info("🟢 【安全】検証モード(IS_PRODUCTION=False)で起動しています。シミュレーターで動作します。")

    try:
        # 1. 起動前のお掃除
        kill_kabu_station()

        # 🔥 変更：2. 最新のAIモデルをKaggleからダウンロード
        logger.info("\n▶️ [STEP 1] 最新AIモデルのダウンロード (Kaggle API)...")
        download_models_from_kaggle()

        # 3. 自動ログインの実行
        logger.info("\n▶️ [STEP 2] auto_login.py を実行...")
        subprocess.run([sys.executable, "auto_login.py"], check=True)
        
        time.sleep(10)

        # 4. AI予測バッチの実行 (シグナル生成)
        logger.info("\n▶️ [STEP 3] daily_batch.py の実行判定...")
        run_batch = True
        
        if is_production:
            from datetime import datetime
            now = datetime.now()
            if now.weekday() >= 5: 
                logger.info("💤 本日は休日（土日）のため、バッチ処理と自動売買をスキップします。")
                run_batch = False

        if run_batch:
            logger.info("🤖 本日のAI予測バッチを実行します。時間がかかる場合があります...")
            subprocess.run([sys.executable, "daily_batch.py"], check=True)

        # 5. 自動売買エンジンの実行（発注・監視）
        logger.info("\n▶️ [STEP 4] auto_trade.py を実行...")
        subprocess.run([sys.executable, "auto_trade.py"], check=True)

    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ スクリプト実行エラー: {e}")
        logger.error(f"   実行コマンド: {' '.join(e.cmd)}")
        if e.stdout: logger.error(f"   標準出力: {e.stdout}")
        if e.stderr: logger.error(f"   標準エラー: {e.stderr}")
    except Exception as e:
        logger.error(f"\n❌ 予期せぬエラーが発生しました: {e}")
    finally:
        logger.info("\n=========================================")
        logger.info("🏁 統合ランナーの処理がすべて完了しました")
        logger.info("=========================================")

if __name__ == "__main__":
    main()