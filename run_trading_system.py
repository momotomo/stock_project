import sys
import time
import subprocess
import psutil
from datetime import datetime
import jpholiday
import logging

# =========================================================
# 自動取引システム 統合実行ランナー
# ---------------------------------------------------------
# 【役割】
# 1. 営業日（平日かつ祝日以外）かどうかの判定
# 2. auto_login.py の実行（ログイン）
# 3. daily_batch.py の実行（AIシグナル作成）
# 4. auto_trade.py の実行（自動発注と監視）
# 5. すべて完了後、kabuステーションを強制終了してメモリを解放
# =========================================================

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def is_market_open():
    """今日が市場の営業日（平日かつ祝日ではない）かどうかを判定"""
    now = datetime.now()
    
    # 1. 土日判定 (0:月, 1:火, 2:水, 3:木, 4:金, 5:土, 6:日)
    if now.weekday() >= 5:
        return False
        
    # 2. 日本の祝日・振替休日の判定
    if jpholiday.is_holiday(now):
        return False
        
    # 3. 年末年始（12/31〜1/3）の特別休場判定
    if (now.month == 12 and now.day == 31) or (now.month == 1 and now.day <= 3):
        return False
        
    return True

def kill_kabu_station():
    """kabuステーションのプロセスを探して強制終了する"""
    logger.info("🧹 kabuステーションのプロセスを確認・終了します...")
    killed = False
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == 'kabu.station.exe':
            proc.kill()
            killed = True
            
    if killed:
        logger.info("✅ kabuステーションを安全に終了しました。")
        time.sleep(3) # 完全に落ちるまで少し待つ
    else:
        logger.info("ℹ️ 終了すべきkabuステーションのプロセスは見つかりませんでした。")

def main():
    logger.info("=========================================")
    logger.info("🌅 自動取引システム 統合ランナー起動")
    logger.info("=========================================")
    
    # --- 休場日チェック ---
    if not is_market_open():
        logger.info("😴 本日は市場の休場日（土日・祝日・年末年始）です。")
        logger.info("システムを起動せずにこのまま終了します。ゆっくりお休みください！")
        return

    logger.info("🟢 本日は営業日です。システム起動シーケンスを開始します。")

    try:
        # 1. 念のため、昨日の残骸（既存のkabuステーション）があれば落とす
        kill_kabu_station()

        # 2. 自動ログインの実行
        logger.info("\n▶️ [STEP 1] auto_login.py を実行します...")
        subprocess.run([sys.executable, "auto_login.py"], check=True)

        # 3. AI予測バッチの実行 (シグナル生成)
        logger.info("\n▶️ [STEP 2] daily_batch.py (AIシグナル生成) を実行します...")
        subprocess.run([sys.executable, "daily_batch.py"], check=True)

        # 4. 自動売買エンジンの実行
        logger.info("\n▶️ [STEP 3] auto_trade.py (自動発注・監視) を実行します...")
        subprocess.run([sys.executable, "auto_trade.py"], check=True)

    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ スクリプトの実行中にエラーが発生して停止しました: {e}")
    except Exception as e:
        logger.error(f"\n❌ 予期せぬエラーが発生しました: {e}")
    finally:
        # 5. エラーが起きても起きなくても、最後に必ずkabuステーションを終了する
        logger.info("\n🏁 本日の全タスクが終了しました。お片付けをします。")
        kill_kabu_station()
        logger.info("=== 統合ランナーの処理が全て完了しました ===")

if __name__ == "__main__":
    main()