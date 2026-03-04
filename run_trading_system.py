import sys
import time
import subprocess
import psutil
import os
import yaml
from datetime import datetime
import jpholiday
import logging

# =========================================================
# 自動取引システム 統合実行ランナー (プロセス名修正版)
# ---------------------------------------------------------
# 【役割】
# 1. 営業日判定（テストモード時は休日でも続行）
# 2. auto_login.py / daily_batch.py / auto_trade.py の順次実行
# 3. 終了後、kabuステーション(KabuS.exe)を確実にクローズ
# =========================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def is_market_open():
    """今日が市場の営業日（平日かつ祝日以外）かどうかを判定"""
    now = datetime.now()
    if now.weekday() >= 5: return False # 土日
    if jpholiday.is_holiday(now): return False # 祝日
    if (now.month == 12 and now.day == 31) or (now.month == 1 and now.day <= 3): return False # 年末年始
    return True

def kill_kabu_station():
    """kabuステーションのプロセス(KabuS.exe)を確実に終了させる"""
    logger.info("🧹 kabuステーションのプロセスを確認・終了します...")
    # 最新バージョンのプロセス名「KabuS.exe」に対応
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

# 🔥 追加：GitHubから最新のモデルをダウンロードする関数
def git_pull_latest():
    """GitHubからKaggleが学習した最新のAIモデルをダウンロードする"""
    logger.info("🌐 GitHubから最新のAIモデルを受信 (git pull) します...")
    try:
        # git pull を実行し、結果を取得する
        result = subprocess.run(["git", "pull"], capture_output=True, text=True, check=True)
        logger.info(f"✅ モデルの同期完了:\n{result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Git Pull エラー (ローカルの変更が衝突している可能性があります):\n{e.stderr.strip()}")
    except Exception as e:
        logger.warning(f"⚠️ Gitコマンドが実行できませんでした (Gitがインストールされていないか、リポジトリ外です): {e}")

def main():
    logger.info("=========================================")
    logger.info("🌅 自動取引システム 統合ランナー起動")
    logger.info("=========================================")
    
    # settings.yml から本番/検証モードを読み取る
    is_production = False
    if os.path.exists("settings.yml"):
        try:
            with open("settings.yml", "r", encoding="utf-8") as f:
                conf = yaml.safe_load(f)
                is_production = conf.get("IS_PRODUCTION", False)
        except Exception as e:
            logger.warning(f"⚠️ 設定ファイルの読み取りに失敗しました: {e}")

    # --- 休場日チェック ---
    if not is_market_open():
        if is_production:
            logger.info("😴 本日は市場の休場日です。本番モードのため、安全のため終了します。")
            return
        else:
            logger.info("🧪 本日は休場日ですが、テストモード(IS_PRODUCTION: false)のため実行を継続します。")
    else:
        logger.info("🟢 本日は営業日です。システムを通常稼働させます。")

    try:
        # 1. 起動前のお掃除（昨日の残骸があれば落とす）
        kill_kabu_station()

        # 🔥 追加：2. 最新のAIモデルをダウンロード
        logger.info("\n▶️ [STEP 1] 最新AIモデルのダウンロード (git pull)...")
        git_pull_latest()

        # 3. 自動ログインの実行 (STEP番号をずらします)
        logger.info("\n▶️ [STEP 2] auto_login.py を実行...")
        subprocess.run([sys.executable, "auto_login.py"], check=True)
        
        # ログイン完了後、APIサーバーが安定するまで少し待機
        time.sleep(10)

        # 4. AI予測バッチの実行 (シグナル生成)
        logger.info("\n▶️ [STEP 3] daily_batch.py の実行判定...")
        if os.path.exists(rec_file):
            mtime = os.path.getmtime(rec_file)
            mdate = datetime.fromtimestamp(mtime).date()
            if mdate == datetime.now().date():
                logger.info(f"✅ 本日のAI予測データ({rec_file})は作成済みのため、AI予測バッチ処理をスキップします。")
                run_batch = False
        
        if run_batch:
            logger.info("🤖 本日のAI予測バッチを実行します。時間がかかる場合があります...")
            subprocess.run([sys.executable, "daily_batch.py"], check=True)

        # 5. 自動売買エンジンの実行（発注・監視）
        logger.info("\n▶️ [STEP 4] auto_trade.py を実行...")
        subprocess.run([sys.executable, "auto_trade.py"], check=True)

    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ スクリプト実行エラー: {e}")
    except Exception as e:
        logger.error(f"\n❌ 予期せぬエラーが発生しました: {e}")
    finally:
        # 5. 最後にお片付け（kabuステーションを終了してメモリを解放）
        logger.info("\n🏁 本日の全タスクが完了しました。お片付けをします。")
        kill_kabu_station()
        logger.info("=== 統合ランナー終了 ===")

if __name__ == "__main__":
    main()