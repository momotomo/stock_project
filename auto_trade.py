import asyncio
import time
import logging
import aiohttp
import csv
import os
from dataclasses import dataclass
from typing import Dict, List

# =========================================================
# kabuステーション 自動取引エンジン (auto_trade.py) - AI連動版
# ---------------------------------------------------------
# 【特徴】
# 1. AI予測結果 (recommendations.csv) の自動読み込みとシグナル判定
# 2. TokenBucketによる「秒間5件」のAPI流量制限の厳密な統制
# 3. インメモリストートマシンによる自律的な価格監視とバリア決済
# =========================================================

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- システム設定 (Config) ---
class Config:
    # ⚠️ 動作環境モード (Trueで本番口座、Falseでシミュレーター環境)
    IS_PRODUCTION = False  
    PORT = 18080 if IS_PRODUCTION else 18081
    API_URL = f"http://localhost:{PORT}/kabusapi"
    
    # 認証情報（ご自身のパスワードに書き換えてください）
    API_PASSWORD = "1111111111"   # kabuステーションAPIのパスワード
    TRADE_PASSWORD = "Tr7smv_jnxg" # 注文・決済パスワード

    # ロット管理とシグナル設定
    MAX_POSITIONS = 1             # 同時に保有する最大銘柄数
    LOT_CALC_MODE = "FIXED"
    FIXED_LOT_SIZE = 100          # 1回の購入株数（1単元）
    ENTRY_THRESHOLD_PROB = 55.0   # 💡 買いシグナルを発火する「明日の上昇確率」のしきい値(%)

    # トリプルバリア設定（利確・損切）
    TAKE_PROFIT_PCT = 0.05        # 利確 5% (0.05)
    STOP_LOSS_PCT = 0.05          # 損切 5% (0.05)
    EXCHANGE = 1                  # 1: 東証

# --- トークンバケット（秒間5件のAPI制限を克服するアルゴリズム） ---
class TokenBucket:
    def __init__(self, capacity: int, fill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.fill_rate = fill_rate
        self.last_fill = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, amount: int = 1):
        """APIを叩く前に必ずこれを呼び、トークンを消費する（無ければ待つ）"""
        async with self.lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_fill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
                self.last_fill = now

                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                await asyncio.sleep(0.1)

# --- APIクライアント ---
class KabuAPI:
    def __init__(self, config: Config):
        self.config = config
        self.token = None
        self.session = None
        self.bucket = TokenBucket(capacity=5, fill_rate=5.0)

    async def start_session(self):
        self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()

    async def _request(self, method: str, endpoint: str, data: dict = None, params: dict = None) -> dict:
        await self.bucket.consume()
        url = f"{self.config.API_URL}/{endpoint}"
        headers = {"X-API-KEY": self.token} if self.token else {}
        
        try:
            async with self.session.request(method, url, headers=headers, json=data, params=params) as response:
                # 💡 HTTPエラー時に、kabuステーションからの詳しいエラーメッセージ（日本語）をログに出すように強化
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API HTTP Error {response.status} at {endpoint}: {error_text}")
                
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"API Request Exception ({endpoint}): {e}")
            return None

    async def get_token(self):
        logger.info(f"APIトークンを取得します (ポート: {self.config.PORT})...")
        data = {"APIPassword": self.config.API_PASSWORD}
        res = await self._request("POST", "token", data=data)
        if res and "Token" in res:
            self.token = res["Token"]
            logger.info("✅ トークン取得成功")
        else:
            logger.error("❌ トークン取得失敗")

    async def get_board(self, symbol: str, exchange: int):
        endpoint = f"board/{symbol}@{exchange}"
        return await self._request("GET", endpoint)

    async def send_order(self, symbol: str, side: str, qty: int, price: float = 0):
        # 💡 API側の非常に厳格なJSON構造チェックを通過させるための「完全版」注文データ
        order_data = {
            "Password": self.config.TRADE_PASSWORD,
            "Symbol": str(symbol),
            "Exchange": int(self.config.EXCHANGE),
            "SecurityType": 1,
            "Side": str(side),
            "CashMargin": 3,         # 3: 現物取引
            "MarginTradeType": 1,    # 💡 必須ダミー: 現物でも省略不可（APIのエラー回避）
            "MarginPremiumUnit": 1,  # 💡 必須ダミー: 同上
            "DelivType": 2 if side == "2" else 0, # 💡 買=2(お預り金), 売=0(指定なし)
            "FundType": "  ",        # 現物保護預りを示す半角スペース2つ
            "AccountType": 4,        # 4: 特定口座
            "Qty": int(qty),
            "Price": 0 if price == 0 else float(price), # 成行の「0」は整数で渡す
            "ExpireDay": 0,
            "FrontOrderType": 10     # 10: 成行
        }
        action = "買" if side == "2" else "売"
        logger.info(f"🚀 発注要求: {action} {symbol} {qty}株 (成行)")
        return await self._request("POST", "sendorder", data=order_data)

# --- インメモリストートマシン（保有状態とバリア監視） ---
@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    take_profit_price: float
    stop_loss_price: float

class PortfolioManager:
    def __init__(self, config: Config, api: KabuAPI):
        self.config = config
        self.api = api
        self.positions: Dict[str, Position] = {}

    def add_position(self, symbol: str, qty: int, entry_price: float):
        tp = entry_price * (1 + self.config.TAKE_PROFIT_PCT)
        sl = entry_price * (1 - self.config.STOP_LOSS_PCT)
        self.positions[symbol] = Position(symbol, qty, entry_price, tp, sl)
        logger.info(f"💼 ポジション追加: {symbol} {qty}株 (取得単価: {entry_price:,.1f}円)")
        logger.info(f"🎯 バリア設定 -> 利確: {tp:,.1f}円 / 損切: {sl:,.1f}円")

    async def check_barriers(self):
        for symbol, pos in list(self.positions.items()):
            board = await self.api.get_board(symbol, self.config.EXCHANGE)
            if not board: continue
            
            current_price = board.get("CurrentPrice") or board.get("PreviousClose")
            if current_price is None:
                continue

            logger.info(f"👀 {symbol} 監視中... 現在値:{current_price:,.1f}円 (利確目標:{pos.take_profit_price:,.1f}円 / 損切防衛:{pos.stop_loss_price:,.1f}円)")

            if current_price >= pos.take_profit_price:
                logger.warning(f"📈 利確バリア到達！ {symbol} の決済（売り）を実行します。")
                await self.execute_exit(pos)
            elif current_price <= pos.stop_loss_price:
                logger.error(f"📉 損切バリア到達！ {symbol} の決済（売り）を実行します。")
                await self.execute_exit(pos)

    async def execute_exit(self, pos: Position):
        res = await self.api.send_order(pos.symbol, side="1", qty=pos.qty)
        if res and res.get("Result") == 0:
            logger.info(f"✅ 決済注文受付成功: {pos.symbol}")
            del self.positions[pos.symbol]
        else:
            logger.error(f"❌ 決済注文失敗: {res}")

# --- AI連携モジュール ---
def get_ticker_mapping() -> dict:
    """銘柄名（日本語）から証券コード（4桁）に変換する辞書を作成"""
    mapping = {}
    pool = {
        "トヨタ自動車": "7203.T", "ソニーG": "6758.T", "三菱UFJ": "8306.T",
        "キーエンス": "6861.T", "NTT": "9432.T", "ファーストリテイリング": "9983.T",
        "東京エレクトロン": "8035.T", "信越化学": "4063.T", "三井住友FG": "8316.T",
        "日立製作所": "6501.T", "伊藤忠商事": "8001.T", "KDDI": "9433.T",
        "ホンダ": "7267.T", "三菱商事": "8058.T", "ソフトバンクG": "9984.T",
        "任天堂": "7974.T", "オリックス": "8591.T"
    }
    # ユーザーのカスタムリストがあれば結合
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r", encoding="utf-8") as f:
            for line in f:
                if ',' in line:
                    name, tk = line.split(',', 1)
                    pool[name.strip()] = tk.strip()

    # ".T" を取り除いて4桁の数字だけにする
    for name, tk in pool.items():
        mapping[name] = tk.replace(".T", "").strip()
    return mapping

def load_ai_signals(config: Config) -> List[dict]:
    """recommendations.csv を読み込み、条件を満たすエントリー候補を返す"""
    signals = []
    if not os.path.exists('recommendations.csv'):
        logger.warning("recommendations.csv が見つかりません。先にAI分析バッチを実行してください。")
        return signals

    mapping = get_ticker_mapping()

    with open('recommendations.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prob = float(row.get('明日の上昇確率', 0))
            if prob >= config.ENTRY_THRESHOLD_PROB:
                name = row.get('銘柄名', '')
                code = mapping.get(name)
                if code:
                    signals.append({
                        'name': name,
                        'symbol': code,
                        'prob': prob
                    })

    # 確率が高い順に並び替え、最大保有銘柄数 (MAX_POSITIONS) までに絞り込む
    signals = sorted(signals, key=lambda x: x['prob'], reverse=True)
    return signals[:config.MAX_POSITIONS]

# --- メインエンジンループ ---
async def main():
    logger.info("=== 自動取引エンジン起動 (AI連動モード) ===")
    config = Config()
    
    if not config.IS_PRODUCTION:
        logger.info("🧪 [シミュレーターモード] ポート18081(検証環境)で動作します。")

    api = KabuAPI(config)
    await api.start_session()
    await api.get_token()

    if not api.token:
        logger.error("APIトークンが取得できなかったため、システムを終了します。")
        await api.close_session()
        return

    portfolio = PortfolioManager(config, api)

    try:
        # 1. AIシグナルの取得と新規発注
        logger.info("=== AI予測データの読み込みとエントリー判定 ===")
        signals = load_ai_signals(config)

        if not signals:
            logger.info(f"😴 本日は「明日の上昇確率 {config.ENTRY_THRESHOLD_PROB}% 以上」の条件を満たす銘柄はありませんでした。エントリーを見送ります。")
        else:
            for sig in signals:
                logger.info(f"🌟 AIシグナル発火！ 銘柄: {sig['name']} ({sig['symbol']}) - 上昇確率: {sig['prob']}%")
                
                # 現在の価格を取得
                board = await api.get_board(sig['symbol'], config.EXCHANGE)
                current_price = None
                if board:
                    current_price = board.get("CurrentPrice") or board.get("PreviousClose")
                    if board.get("CurrentPrice") is None:
                        logger.warning("🌙 市場時間外のため前日終値を参照します。")
                
                if not current_price:
                    current_price = 2000.0 # フェイルセーフ用のダミー価格
                    logger.warning("⚠️ 価格が取得できないためダミー価格(2000円)を適用します。")

                # API経由で買い注文を発射！
                res = await api.send_order(sig['symbol'], side="2", qty=config.FIXED_LOT_SIZE)
                if res and res.get("Result") == 0:
                    logger.info(f"✅ 買い注文の送信に成功しました。")
                else:
                    logger.warning(f"⚠️ 注文送信に失敗したか、仮想発注です。レスポンス: {res}")
                
                # 監視リスト（インメモリ）に追加
                portfolio.add_position(sig['symbol'], config.FIXED_LOT_SIZE, current_price)
        
        # 2. 保有銘柄のリアルタイム監視（バリア決済）
        logger.info("=== リアルタイムインメモリ監視ループ開始 ===")
        while True:
            await portfolio.check_barriers()
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        logger.info("システムを安全に停止します...")
    finally:
        await api.close_session()

if __name__ == "__main__":
    import sys
    # Windows用 asyncio のおまじない
    if sys.platform == "win32" and sys.version_info < (3, 14):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except AttributeError:
            pass
            
    asyncio.run(main())