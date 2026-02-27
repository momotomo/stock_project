import asyncio
import time
import logging
import aiohttp
from dataclasses import dataclass
from typing import Dict

# =========================================================
# kabuステーション 自動取引エンジン (auto_trade.py)
# ---------------------------------------------------------
# 【特徴】
# 1. TokenBucketによる「秒間5件」のAPI流量制限の厳密な統制
# 2. インメモリストートマシンによる自律的な価格監視とバリア決済
# 3. テスト運用向けの安全設計（シミュレーターポート使用）
# =========================================================

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- システム設定 (Config) ---
class Config:
    # ⚠️ 動作環境モード (Trueで本番口座、Falseでシミュレーター環境)
    # 最初は必ず False(18081ポート) で動作テストを行ってください。
    IS_PRODUCTION = False  
    PORT = 18080 if IS_PRODUCTION else 18081
    API_URL = f"http://localhost:{PORT}/kabusapi"
    
    # 認証情報（ご自身のパスワードに書き換えてください）
    API_PASSWORD = "1111111111"   # kabuステーションAPIのパスワード
    TRADE_PASSWORD = "Tr7smv_jnxg" # 注文・決済パスワード

    # ロット管理（Q2, Q3対応）
    MAX_POSITIONS = 1
    LOT_CALC_MODE = "FIXED"
    FIXED_LOT_SIZE = 100  # 1回の購入株数（1単元）

    # トリプルバリア設定（利確・損切）
    ENTRY_THRESHOLD_PROB = 60.0 # 買いエントリのしきい値(%)
    TAKE_PROFIT_PCT = 0.05      # 利確 5% (0.05)
    STOP_LOSS_PCT = 0.05        # 損切 5% (0.05)

    # 対象銘柄（最初は1銘柄でテスト）
    TARGET_SYMBOL = "7203"  # トヨタ自動車
    EXCHANGE = 1  # 1: 東証

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
                # トークンが回復するまで少し待機
                await asyncio.sleep(0.1)

# --- APIクライアント ---
class KabuAPI:
    def __init__(self, config: Config):
        self.config = config
        self.token = None
        self.session = None
        # kabuステーションの制限: 秒間5リクエストを上限とする
        self.bucket = TokenBucket(capacity=5, fill_rate=5.0)

    async def start_session(self):
        self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()

    async def _request(self, method: str, endpoint: str, data: dict = None, params: dict = None) -> dict:
        # 通信前にトークンバケットをチェックし、制限超過を防ぐ
        await self.bucket.consume()
        
        url = f"{self.config.API_URL}/{endpoint}"
        headers = {}
        if self.token:
            headers["X-API-KEY"] = self.token
        
        try:
            async with self.session.request(method, url, headers=headers, json=data, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"API Request Error ({endpoint}): {e}")
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
        # 銘柄の現在価格を取得 (kabuステーション特有の 銘柄@市場 形式)
        endpoint = f"board/{symbol}@{exchange}"
        return await self._request("GET", endpoint)

    async def send_order(self, symbol: str, side: str, qty: int, price: float = 0):
        # side: "1" (売), "2" (買)
        # price: 0 (成行)
        order_data = {
            "Password": self.config.TRADE_PASSWORD,
            "Symbol": symbol,
            "Exchange": self.config.EXCHANGE,
            "SecurityType": 1,
            "Side": side,
            "CashMargin": 1, # 現物
            "MarginTradeType": 1,
            "DelivType": 2,
            "AccountType": 4, # 特定口座
            "Qty": qty,
            "Price": price,
            "ExpireDay": 0,
            "FrontOrderType": 10 # 成行
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
        self.positions: Dict[str, Position] = {} # メモリ上で保有状態を管理

    def add_position(self, symbol: str, qty: int, entry_price: float):
        tp = entry_price * (1 + self.config.TAKE_PROFIT_PCT)
        sl = entry_price * (1 - self.config.STOP_LOSS_PCT)
        self.positions[symbol] = Position(symbol, qty, entry_price, tp, sl)
        logger.info(f"💼 ポジション追加: {symbol} {qty}株 (取得単価: {entry_price:,.1f}円)")
        logger.info(f"🎯 バリア設定 -> 利確: {tp:,.1f}円 / 損切: {sl:,.1f}円")

    async def check_barriers(self):
        """毎ループで現在価格を取得し、トリプルバリアを自律的に発動させる"""
        for symbol, pos in list(self.positions.items()):
            board = await self.api.get_board(symbol, self.config.EXCHANGE)
            if not board or "CurrentPrice" not in board:
                continue
            
            current_price = board["CurrentPrice"]
            if current_price is None:
                continue

            logger.info(f"👀 {symbol} 監視中... 現在値:{current_price:,.1f}円 (利確目標:{pos.take_profit_price:,.1f}円 / 損切防衛:{pos.stop_loss_price:,.1f}円)")

            # バリア判定
            if current_price >= pos.take_profit_price:
                logger.warning(f"📈 利確バリア到達！ {symbol} の決済（売り）を実行します。")
                await self.execute_exit(pos)
            elif current_price <= pos.stop_loss_price:
                logger.error(f"📉 損切バリア到達！ {symbol} の決済（売り）を実行します。")
                await self.execute_exit(pos)

    async def execute_exit(self, pos: Position):
        # 決済注文（売り成行）を送信
        res = await self.api.send_order(pos.symbol, side="1", qty=pos.qty)
        if res and res.get("Result") == 0:
            logger.info(f"✅ 決済注文受付成功: {pos.symbol}")
            del self.positions[pos.symbol] # メモリから削除
        else:
            logger.error(f"❌ 決済注文失敗: {res}")

# --- メインエンジンループ ---
async def main():
    logger.info("=== 自動取引エンジン起動 ===")
    config = Config()
    
    if not config.IS_PRODUCTION:
        logger.info("🧪 [シミュレーターモード] ポート18081(検証環境)で動作します。実際の資金は動きません。")

    api = KabuAPI(config)
    await api.start_session()
    await api.get_token()

    if not api.token:
        logger.error("APIトークンが取得できなかったため、システムを終了します。")
        await api.close_session()
        return

    portfolio = PortfolioManager(config, api)

    try:
        # =========================================================
        # 【テストフェーズ】
        # 本来はここで AI (app.py) の推論結果を読み込み、スコアが高ければエントリーします。
        # 今回は「エンジンの動作確認」のため、起動時に強制的に1単元買ったと仮定し、
        # 監視・バリア決済のループが正しく動くかテストします。
        # =========================================================
        logger.info("=== 仮想エントリー処理開始 ===")
        board = await api.get_board(config.TARGET_SYMBOL, config.EXCHANGE)
        if board and board.get("CurrentPrice"):
            current_price = board["CurrentPrice"]
            # 実際のAPI買い発注はコメントアウトし、ポジションだけ追加します（安全のため）
            # await api.send_order(config.TARGET_SYMBOL, side="2", qty=config.FIXED_LOT_SIZE)
            portfolio.add_position(config.TARGET_SYMBOL, config.FIXED_LOT_SIZE, current_price)
        
        logger.info("=== リアルタイムインメモリ監視ループ開始 ===")
        while True:
            # 設定したバリア（利確・損切）に触れていないか監視
            await portfolio.check_barriers()
            
            # APIの負荷を減らすため、数秒おきにチェック
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        logger.info("システムを安全に停止します...")
    finally:
        await api.close_session()

if __name__ == "__main__":
    # Windowsでasyncioを安定動作させるための設定
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 実行前に必要なライブラリ(aiohttp)がインストールされているか確認してください
    # pip install aiohttp
    asyncio.run(main())