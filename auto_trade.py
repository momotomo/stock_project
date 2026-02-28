import asyncio
import time
import logging
import aiohttp

# =========================================================
# kabuステーション 自動取引テストスクリプト (現物 買い→売り 連続テスト版)
# ---------------------------------------------------------
# 指定した銘柄を買い、10秒後にすぐ売る「往復テスト」を行います。
# =========================================================

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class Config:
    IS_PRODUCTION = False  
    PORT = 18080 if IS_PRODUCTION else 18081
    API_URL = f"http://localhost:{PORT}/kabusapi"
    
    # 認証情報（ご自身のパスワードに書き換えてください）
    API_PASSWORD = "1111111111"     # kabuステーションAPI(検証用)のパスワード
    TRADE_PASSWORD = "Tr7smv_jnxg" # 取引(注文)パスワード
    EXCHANGE = 1  # 1: 東証

    # 💡 【追加】現物・信用の切り替えスイッチ
    # "CASH" = 現物取引 / "MARGIN" = 信用取引（制度信用）
    TRADE_MODE = "CASH" 

class TokenBucket:
    def __init__(self, capacity: int, fill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.fill_rate = fill_rate
        self.last_fill = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, amount: int = 1):
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

    async def send_order(self, symbol: str, side: str, qty: int, price: float = 0, is_close: bool = False, hold_id: str = None):
        """ 発注処理（現物・信用を自動で出し分け） """
        
        if self.config.TRADE_MODE == "CASH":
            # --- 現物取引のパラメータ ---
            cash_margin = 1
            margin_trade_type = 1 # ダミー値
            deliv_type = 2 if side == "2" else 0
            fund_type = "02" if side == "2" else "  "
        else:
            # --- 信用取引のパラメータ ---
            cash_margin = 3 if is_close else 2 # 3:返済, 2:新規
            margin_trade_type = 1 # 1:制度信用
            deliv_type = 0
            fund_type = "  "

        order_data = {
            "Password": self.config.TRADE_PASSWORD,
            "Symbol": str(symbol),
            "Exchange": int(self.config.EXCHANGE),
            "SecurityType": 1,
            "Side": str(side),
            "CashMargin": cash_margin,
            "MarginTradeType": margin_trade_type, 
            "MarginPremiumUnit": 1, 
            "DelivType": deliv_type,
            "FundType": fund_type, 
            "AccountType": 4,
            "Qty": int(qty),
            "Price": int(price) if price == 0 else float(price),
            "ExpireDay": 0,
            "FrontOrderType": 10
        }

        # 💡 信用返済の場合は、どの借金(建玉)を返すかの指定が必要
        if self.config.TRADE_MODE == "MARGIN" and is_close:
            if hold_id:
                order_data["ClosePositions"] = [{"HoldID": hold_id, "Qty": int(qty)}]
            else:
                logger.warning("⚠️ 信用返済ですが建玉ID(HoldID)が指定されていません（APIで弾かれる可能性大）")

        action = "買" if side == "2" else "売"
        trade_type_str = "現物" if self.config.TRADE_MODE == "CASH" else ("信用返済" if is_close else "信用新規")
        logger.info(f"🚀 発注リクエスト送信 [{trade_type_str}]: {action} {symbol} {qty}株 (成行)")
        
        return await self._request("POST", "sendorder", data=order_data)

async def main():
    logger.info("=== 自動取引エンジン起動 (売買往復テストモード) ===")
    config = Config()
    
    if not config.IS_PRODUCTION:
        logger.info("🧪 [シミュレーターモード] ポート18081(検証環境)で動作します。")

    api = KabuAPI(config)
    await api.start_session()
    await api.get_token()

    if not api.token:
        logger.error("APIトークンが取得できなかったため終了します。")
        await api.close_session()
        return

    try:
        # =========================================================
        # 【発注テストシーケンス】買い注文 → 10秒待機 → 売り(決済)注文
        # =========================================================
        test_symbol = "8591" # テスト銘柄：オリックス
        qty = 100
        
        logger.info(f"=== 🟢 テストステップ1: エントリー注文 ({test_symbol}) ===")
        # is_close=False (新規注文) として送信
        res_buy = await api.send_order(test_symbol, side="2", qty=qty, is_close=False)
        
        if res_buy and res_buy.get("Result") == 0:
            logger.info(f"✅ エントリー注文 成功！ 受付番号: {res_buy.get('OrderId')}")
        else:
            logger.error(f"❌ エントリー注文 失敗。処理を中断します。")
            return
            
        logger.info("⏳ 注文の約定とシステムの反映を待つため、10秒間待機します...")
        await asyncio.sleep(10)
        
        logger.info(f"=== 🔴 テストステップ2: エグジット(決済)注文 ({test_symbol}) ===")
        # is_close=True (決済注文) として送信
        res_sell = await api.send_order(test_symbol, side="1", qty=qty, is_close=True)
        
        if res_sell and res_sell.get("Result") == 0:
            logger.info(f"✅ 決済注文 成功！ 受付番号: {res_sell.get('OrderId')}")
        else:
            logger.error(f"❌ 決済注文 失敗。")
            
        logger.info("🎉 買い・売り両方のテストシーケンスが完了しました。")

    except KeyboardInterrupt:
        logger.info("システムを安全に停止します...")
    finally:
        await api.close_session()

if __name__ == "__main__":
    import sys
    if sys.platform == "win32" and sys.version_info < (3, 14):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except AttributeError:
            pass
            
    asyncio.run(main())