import asyncio
import time
import logging
import aiohttp
import csv
import os
import yaml
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List

# =========================================================
# kabuステーション 自動取引エンジン (auto_trade.py) - 統合本番仕様
# =========================================================

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- システム設定 (Config) ---
class Config:
    def __init__(self):
        self.config_file = "settings.yml"
        self.load_config()

    def load_config(self):
        default_config = {
            "IS_PRODUCTION": False,
            "API_PASSWORD": "YOUR_API_PASSWORD",
            "TRADE_PASSWORD": "YOUR_TRADE_PASSWORD",
            "EXCHANGE": 1,
            "TRADE_STYLE": "day",             # day: 14:50強制決済, swing: 持ち越し
            "TARGET_HORIZON": "明日",         # 明日, 1W, 2W, 1M, 3M, 6M, 1Y
            "TRADE_MODE": "CASH",
            "FUND_TYPE": "21",                # デフォルトは特定口座(21)
            "MAX_POSITIONS": 1,
            "LOT_CALC_MODE": "AUTO",
            "FIXED_LOT_SIZE": 100,
            "AUTO_INVEST_RATIO": 0.3,
            "ENTRY_THRESHOLD_PROB": 55.0,
            "TAKE_PROFIT_PCT": 0.05,
            "STOP_LOSS_PCT": 0.05
        }

        if not os.path.exists(self.config_file):
            config_data = default_config
        else:
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                logger.info(f"⚙️ 設定ファイル '{self.config_file}' を読み込みました。")
            except Exception as e:
                logger.error(f"❌ 設定ファイルの読み込みに失敗しました。デフォルト設定を使用します: {e}")
                config_data = default_config

        self.IS_PRODUCTION = config_data.get("IS_PRODUCTION", default_config["IS_PRODUCTION"])
        self.PORT = 18080 if self.IS_PRODUCTION else 18081
        self.API_URL = f"http://localhost:{self.PORT}/kabusapi"
        
        self.API_PASSWORD = str(config_data.get("API_PASSWORD", default_config["API_PASSWORD"]))
        self.TRADE_PASSWORD = str(config_data.get("TRADE_PASSWORD", default_config["TRADE_PASSWORD"]))
        self.EXCHANGE = config_data.get("EXCHANGE", default_config["EXCHANGE"])
        
        self.TRADE_STYLE = config_data.get("TRADE_STYLE", default_config["TRADE_STYLE"])
        self.TARGET_HORIZON = str(config_data.get("TARGET_HORIZON", default_config["TARGET_HORIZON"]))

        self.TRADE_MODE = config_data.get("TRADE_MODE", default_config["TRADE_MODE"])
        # 🔥 修正: FUND_TYPEは廃止し、ACCOUNT_TYPE(特定口座=4)を読み込む
        self.ACCOUNT_TYPE = int(config_data.get("ACCOUNT_TYPE", 4))
        
        self.MAX_POSITIONS = config_data.get("MAX_POSITIONS", default_config["MAX_POSITIONS"])
        self.LOT_CALC_MODE = config_data.get("LOT_CALC_MODE", default_config["LOT_CALC_MODE"])
        self.FIXED_LOT_SIZE = config_data.get("FIXED_LOT_SIZE", default_config["FIXED_LOT_SIZE"])
        self.AUTO_INVEST_RATIO = float(config_data.get("AUTO_INVEST_RATIO", default_config["AUTO_INVEST_RATIO"]))
        
        self.ENTRY_THRESHOLD_PROB = float(config_data.get("ENTRY_THRESHOLD_PROB", default_config["ENTRY_THRESHOLD_PROB"]))
        self.TAKE_PROFIT_PCT = float(config_data.get("TAKE_PROFIT_PCT", default_config["TAKE_PROFIT_PCT"]))
        self.STOP_LOSS_PCT = float(config_data.get("STOP_LOSS_PCT", default_config["STOP_LOSS_PCT"]))

# --- トークンバケット ---
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
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API HTTP Error {response.status} at {endpoint}: {error_text}")
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"API Request Exception ({endpoint}): {e}")
            return None

    async def get_token(self):
        data = {"APIPassword": self.config.API_PASSWORD}
        res = await self._request("POST", "token", data=data)
        if res and "Token" in res:
            self.token = res["Token"]
            logger.info("✅ トークン取得成功")
        else:
            logger.error("❌ トークン取得失敗")

    async def get_board(self, symbol: str, exchange: int):
        return await self._request("GET", f"board/{symbol}@{exchange}")

    async def get_wallet_cash(self):
        """現物買付余力照会"""
        return await self._request("GET", "wallet/cash")

    async def get_wallet_margin(self, symbol: str, exchange: int):
        """信用新規建余力照会"""
        return await self._request("GET", f"wallet/margin/{symbol}@{exchange}")

    async def get_positions(self, product: int = 0):
        """保有ポジション(建玉)一覧を取得。product: 1=現物, 2=信用"""
        return await self._request("GET", "positions", params={"product": product})

    async def send_order(self, symbol: str, side: str, qty: int, price: float = 0, is_close: bool = False, hold_id: str = None):
        if self.config.TRADE_MODE == "CASH":
            cash_margin = 1
            margin_trade_type = 1 
            deliv_type = 2 if side == "2" else 0
            
            # 🔥 修正: API仕様書に基づき、買い=02、売り=空白2文字に変更
            if side == "2":
                fund_type = "02"  # 買いの時は保護(02)
            else:
                fund_type = "  "  # 売りの時は指定なし(空白2文字)
                
        else:
            cash_margin = 3 if is_close else 2
            margin_trade_type = 1 
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
            "AccountType": self.config.ACCOUNT_TYPE,
            "Qty": int(qty),
            "Price": int(price) if price == 0 else float(price),
            "ExpireDay": 0,
            "FrontOrderType": 10
        }

        if self.config.TRADE_MODE == "MARGIN" and is_close:
            if hold_id:
                order_data["ClosePositions"] = [{"HoldID": hold_id, "Qty": int(qty)}]
            else:
                logger.warning("⚠️ 信用返済ですが建玉ID(HoldID)が指定されていません")

        action = "買" if side == "2" else "売"
        trade_type_str = "現物" if self.config.TRADE_MODE == "CASH" else ("信用返済" if is_close else "信用新規")
        
        if not self.config.IS_PRODUCTION:
            logger.info(f"🧪 [シミュレーター] 発注スキップ [{trade_type_str}]: {action} {symbol} {qty}株 (成行)")
            return {"Result": 0, "OrderId": "SIM_" + str(int(time.time()))}

        logger.info(f"🚀 発注リクエスト送信 [{trade_type_str}]: {action} {symbol} {qty}株 (成行)")
        return await self._request("POST", "sendorder", data=order_data)

# --- ポジション管理 ---
@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    take_profit_price: float
    stop_loss_price: float
    hold_id: str = None

class PortfolioManager:
    def __init__(self, config: Config, api: KabuAPI):
        self.config = config
        self.api = api
        self.positions: Dict[str, Position] = {}

    def add_position(self, symbol: str, qty: int, entry_price: float, hold_id: str = None):
        tp = entry_price * (1 + self.config.TAKE_PROFIT_PCT)
        sl = entry_price * (1 - self.config.STOP_LOSS_PCT)
        self.positions[symbol] = Position(symbol, qty, entry_price, tp, sl, hold_id)
        mode_str = "現物" if self.config.TRADE_MODE == "CASH" else "信用"
        logger.info(f"💼 ポジション登録 [{mode_str}]: {symbol} {qty}株 (取得単価: {entry_price:,.1f}円)")
        logger.info(f"🎯 バリア設定 -> 利確: {tp:,.1f}円 / 損切: {sl:,.1f}円")

    async def sync_positions(self):
        logger.info("🔄 前日からの持ち越しポジション（残高）を確認中...")
        product = 1 if self.config.TRADE_MODE == "CASH" else 2
        positions_data = await self.api.get_positions(product=product)
        
        if not positions_data:
            logger.info("保有しているポジションはありませんでした。")
            return

        for p in positions_data:
            symbol = p.get("Symbol")
            qty = p.get("HoldQty", 0)
            entry_price = p.get("Price", 0)
            hold_id = p.get("HoldID")
            
            if qty > 0 and symbol:
                self.add_position(symbol, qty, entry_price, hold_id)
                logger.info(f"📥 既存ポジションを認識しました: {symbol}")

    async def check_barriers(self):
        for symbol, pos in list(self.positions.items()):
            board = await self.api.get_board(symbol, self.config.EXCHANGE)
            if not board: continue
            
            current_price = board.get("CurrentPrice") or board.get("PreviousClose")
            
            if not self.config.IS_PRODUCTION and current_price is None:
                import random
                fluctuation = pos.entry_price * random.uniform(-0.1, 0.1)
                current_price = pos.entry_price + fluctuation

            if current_price is None: continue
            
            logger.info(f"👀 {symbol} 監視中... 現在値:{current_price:,.1f}円 (利確目標:{pos.take_profit_price:,.1f}円 / 損切防衛:{pos.stop_loss_price:,.1f}円)")

            if current_price >= pos.take_profit_price:
                logger.warning(f"📈 利確バリア到達！ {symbol} の決済（売り）を実行します。")
                await self.execute_exit(pos)
            elif current_price <= pos.stop_loss_price:
                logger.error(f"📉 損切バリア到達！ {symbol} の決済（売り）を実行します。")
                await self.execute_exit(pos)

    async def execute_exit(self, pos: Position):
        res = await self.api.send_order(pos.symbol, side="1", qty=pos.qty, is_close=True, hold_id=pos.hold_id)
        if res and res.get("Result") == 0:
            logger.info(f"✅ 決済注文受付成功: {pos.symbol}")
            del self.positions[pos.symbol]
        else:
            logger.error(f"❌ 決済注文失敗: {res}")

# --- AI連携モジュール ---
def get_ticker_mapping() -> dict:
    mapping = {}
    pool = {
        "トヨタ自動車": "7203.T", "ソニーG": "6758.T", "三菱UFJ": "8306.T",
        "キーエンス": "6861.T", "NTT": "9432.T", "ファーストリテイリング": "9983.T",
        "東京エレクトロン": "8035.T", "信越化学": "4063.T", "三井住友FG": "8316.T",
        "日立製作所": "6501.T", "伊藤忠商事": "8001.T", "KDDI": "9433.T",
        "ホンダ": "7267.T", "三菱商事": "8058.T", "ソフトバンクG": "9984.T",
        "任天堂": "7974.T", "オリックス": "8591.T", "ANA": "9202.T", "三井物産": "8031.T",
        "ダイキン工業": "6367.T", "武田薬品": "4502.T", "リクルートHD": "6098.T",
        "みずほFG": "8411.T", "村田製作所": "6981.T", "デンソー": "6902.T",
        "ファナック": "6954.T", "アステラス製薬": "4503.T", "セブン＆アイ": "3382.T",
        "第一三共": "4568.T", "コマツ": "6301.T", "丸紅": "8002.T"
    }
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r", encoding="utf-8") as f:
            for line in f:
                if ',' in line:
                    name, tk = line.split(',', 1)
                    pool[name.strip()] = tk.strip()
    for name, tk in pool.items():
        mapping[name] = tk.replace(".T", "").strip()
    return mapping

def load_ai_signals(config: Config) -> List[dict]:
    signals = []
    if not os.path.exists('recommendations.csv'): return signals
    mapping = get_ticker_mapping()
    with open('recommendations.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        target_col = None
        if config.TARGET_HORIZON == "明日":
            target_col = "明日の上昇確率"
        else:
            for col in reader.fieldnames:
                if col.startswith(f"{config.TARGET_HORIZON} 利確"):
                    target_col = col
                    break
                    
        if not target_col:
            logger.error(f"❌ ターゲット期間 '{config.TARGET_HORIZON}' に対応するデータ列が見つかりません。")
            return signals
            
        logger.info(f"📊 エントリー基準指標: 【 {target_col} 】を使用します。")

        for row in reader:
            try:
                prob = float(row.get(target_col, 0))
            except ValueError:
                prob = 0.0
                
            if prob >= config.ENTRY_THRESHOLD_PROB:
                name = row.get('銘柄名', '')
                code = mapping.get(name)
                if code: signals.append({'name': name, 'symbol': code, 'prob': prob})
                
    signals = sorted(signals, key=lambda x: x['prob'], reverse=True)
    return signals

# --- メインループ ---
async def main():
    config = Config()
    logger.info(f"=== 自動取引エンジン起動 (モード: {config.TRADE_STYLE.upper()}) ===")
    
    api = KabuAPI(config)
    await api.start_session()
    await api.get_token()

    if not api.token: return

    portfolio = PortfolioManager(config, api)
    
    await portfolio.sync_positions()

    try:
        now = datetime.now()
        if config.TRADE_STYLE == "day" and (now.hour > 14 or (now.hour == 14 and now.minute >= 30)):
            logger.info("🕒 14:30を過ぎているため、本日の新規エントリーは見送ります（デイトレモード）。")
            signals = []
        else:
            signals = load_ai_signals(config)

        if not signals:
            logger.info("😴 本日は新規エントリー条件を満たす銘柄はありませんでした。")
        else:
            for sig in signals:
                if sig['symbol'] in portfolio.positions:
                    logger.info(f"🚫 [見送り] {sig['name']} ({sig['symbol']}) は既に保有中のため追加購入しません(重複防止)。")
                    continue
                    
                if len(portfolio.positions) >= config.MAX_POSITIONS:
                    logger.warning(f"🚫 最大保有銘柄数({config.MAX_POSITIONS})に達しているため、新規エントリーを見送ります。")
                    break

                logger.info(f"🌟 AIシグナル発火！ 銘柄: {sig['name']} ({sig['symbol']}) - 確率: {sig['prob']}%")
                
                board = await api.get_board(sig['symbol'], config.EXCHANGE)
                current_price = board.get("CurrentPrice") or board.get("PreviousClose") if board else None
                
                if not current_price:
                    current_price = 2000.0
                    logger.warning("⚠️ 価格が取得できないためダミー価格(2000円)を適用します。")

                qty = 0
                if config.LOT_CALC_MODE == "FIXED":
                    qty = config.FIXED_LOT_SIZE
                elif config.LOT_CALC_MODE == "AUTO":
                    available_cash = 0
                    if config.TRADE_MODE == "CASH":
                        wallet = await api.get_wallet_cash()
                        if wallet: available_cash = wallet.get("StockAccountWallet", 0)
                    else:
                        wallet = await api.get_wallet_margin(sig['symbol'], config.EXCHANGE)
                        if wallet: available_cash = wallet.get("MarginAccountWallet", 0)
                    
                    if not available_cash:
                        available_cash = 1000000

                    budget_per_trade = (available_cash * config.AUTO_INVEST_RATIO) / config.MAX_POSITIONS
                    max_shares = int(budget_per_trade / current_price)
                    qty = (max_shares // 100) * 100 

                    logger.info(f"💰 ロット自動計算: 余力={available_cash:,.0f}円 -> 割当予算={budget_per_trade:,.0f}円 -> 算出ロット={qty}株")

                if qty == 0:
                    logger.warning(f"⚠️ 資金不足（算出ロット0株）のため、{sig['symbol']} のエントリーを見送ります。")
                    continue

                res = await api.send_order(sig['symbol'], side="2", qty=qty, is_close=False)
                
                if res and res.get("Result") == 0:
                    logger.info(f"✅ エントリー注文送信成功")
                    portfolio.add_position(sig['symbol'], qty, current_price, None)
                else:
                    logger.warning(f"⚠️ 注文送信に失敗しました。レスポンス: {res}")
        
        has_force_closed_today = False
        
        if portfolio.positions:
            logger.info("=== ⏱ リアルタイムインメモリ監視ループ開始 ===")
            while True:
                now = datetime.now()
                
                if config.TRADE_STYLE == "day" and now.hour == 14 and now.minute >= 50:
                    if not has_force_closed_today:
                        logger.warning("⏰ 14:50 を回りました！デイトレモードのため、全保有銘柄を強制決済(成行売)します！")
                        for sym, pos in list(portfolio.positions.items()):
                            await portfolio.execute_exit(pos)
                        has_force_closed_today = True
                        logger.info("🏁 本日のデイトレード取引を完全に終了します。お疲れ様でした！")
                    break
                    
                if now.hour >= 15:
                    logger.info("🌙 15:00を回りました。本日の市場監視を終了します。")
                    if portfolio.positions:
                        logger.info(f"📦 現在の持ち越し銘柄数: {len(portfolio.positions)}銘柄 (スイングモードで翌日に持ち越します)")
                    break

                await portfolio.check_barriers()
                
                if not portfolio.positions and config.TRADE_STYLE == "day":
                    logger.info("📉 全てのポジションの決済が完了しました。")
                    break

                await asyncio.sleep(5)

    except KeyboardInterrupt:
        logger.info("システムを停止します...")
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