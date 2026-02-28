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
# 自動取引システム 統合実行ランナー (休日テスト対応版)
# ---------------------------------------------------------
# 【役割】
# 1. 営業日判定（テストモード時は休日でも続行）
# 2. auto_login.py / daily_batch.py / auto_trade.py の順次実行
# 3. 終了後、kabuステーションを確実にクローズ（メモリ解放）
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
    """kabuステーションのプロセスを確実に終了させる"""
    logger.info("🧹 kabuステーションのプロセスを確認・終了します...")
    killed = False
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] == 'kabu.station.exe':
                proc.kill()
                killed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
    if killed:
        logger.info("✅ kabuステーションを終了しました。")
        time.sleep(3)
    else:
        logger.info("ℹ️ 起動中のkabuステーションは見つかりませんでした。")

def main():
    logger.info("=========================================")
    logger.info("🌅 自動取引システム 統合ランナー起動")
    logger.info("=========================================")
    
    # settings.yml を読み込んでモードを確認
    is_production = False
    if os.path.exists("settings.yml"):
        try:
            with open("settings.yml", "r", encoding="utf-8") as f:
                conf = yaml.safe_load(f)
                is_production = conf.get("IS_PRODUCTION", False)
        except Exception as e:
            logger.warning(f"⚠️ 設定ファイルの読み込みに失敗したためデフォルト(検証)で動作します: {e}")

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
        # 1. 起動前のお掃除（多重起動防止）
        kill_kabu_station()

        # 2. 自動ログイン
        logger.info("\n▶️ [STEP 1] auto_login.py を実行...")
        subprocess.run([sys.executable, "auto_login.py"], check=True)

        # 3. AI予測バッチ
        logger.info("\n▶️ [STEP 2] daily_batch.py を実行...")
        subprocess.run([sys.executable, "daily_batch.py"], check=True)

        # 4. 自動売買エンジン（発注・監視）
        logger.info("\n▶️ [STEP 3] auto_trade.py を実行...")
        # auto_trade.py 内に「休日ならダミー価格で動く」ロジックが入っているので安心です
        subprocess.run([sys.executable, "auto_trade.py"], check=True)

    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ スクリプト実行エラー: {e}")
    except Exception as e:
        logger.error(f"\n❌ 予期せぬエラー: {e}")
    finally:
        # 5. 何があっても最後にお片付け（これでアプリの「起動しっぱなし」を防ぎます）
        logger.info("\n🏁 本日の全タスクが完了しました。お片付けをします。")
        kill_kabu_station()
        logger.info("=== 統合ランナー終了 ===")

if __name__ == "__main__":
    main()