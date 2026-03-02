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
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/auto_trade_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
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
            "EXCHANGE": 9,
            "TRADE_STYLE": "day",             
            "TARGET_HORIZON": "明日",         
            "TRADE_MODE": "CASH",
            "ACCOUNT_TYPE": 4,
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
        self.EXCHANGE = int(config_data.get("EXCHANGE", default_config["EXCHANGE"]))
        
        self.TRADE_STYLE = config_data.get("TRADE_STYLE", default_config["TRADE_STYLE"])
        self.TARGET_HORIZON = str(config_data.get("TARGET_HORIZON", default_config["TARGET_HORIZON"]))

        self.TRADE_MODE = config_data.get("TRADE_MODE", default_config["TRADE_MODE"])
        self.ACCOUNT_TYPE = int(config_data.get("ACCOUNT_TYPE", default_config["ACCOUNT_TYPE"]))
        
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
        return await self._request("GET", f"board/{symbol}@1")

    async def get_wallet_cash(self):
        return await self._request("GET", "wallet/cash")

    async def get_wallet_margin(self, symbol: str, exchange: int):
        return await self._request("GET", f"wallet/margin/{symbol}@1")

    async def get_positions(self, product: int = 0):
        return await self._request("GET", "positions", params={"product": product})

    async def get_orders(self, product: int = 0):
        params = {"product": product} if product != 0 else {}
        return await self._request("GET", "orders", params=params)

    async def send_order(self, symbol: str, side: str, qty: int, price: float = 0, is_close: bool = False, hold_id: str = None, exchange: int = None):
        if self.config.TRADE_MODE == "CASH":
            cash_margin = 1
            margin_trade_type = 1 
            deliv_type = 2 if side == "2" else 0
            
            if side == "2":
                fund_type = "AA"  
            else:
                fund_type = "  "  
                
        else:
            cash_margin = 3 if is_close else 2
            margin_trade_type = 1 
            deliv_type = 0
            fund_type = "  "

        target_exchange = exchange if exchange is not None else self.config.EXCHANGE

        order_data = {
            "Password": self.config.TRADE_PASSWORD,
            "Symbol": str(symbol),
            "Exchange": int(target_exchange),
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
            logger.info(f"🧪 [シミュレーター] 発注スキップ [{trade_type_str}]: {action} {symbol} {qty}株 (成行) [市場: {target_exchange}]")
            return {"Result": 0, "OrderId": "SIM_" + str(int(time.time()))}

        logger.info(f"🚀 発注リクエスト送信 [{trade_type_str}]: {action} {symbol} {qty}株 (成行) [市場: {target_exchange}]")
        return await self._request("POST", "sendorder", data=order_data)

# --- 未約定の注文を取得する関数 ---
async def get_active_orders(api: KabuAPI, log: bool = True):
    active_count = 0
    active_symbols = []
    active_details = []
    orders = await api.get_orders()
    if orders:
        for order in orders:
            state = order.get("State")
            order_qty = order.get("OrderQty", 0)
            cum_qty = order.get("CumQty", 0)
            side = order.get("Side")
            
            if state != 5 and cum_qty < order_qty:
                symbol = order.get("Symbol")
                active_count += 1
                if symbol:
                    active_symbols.append(symbol)
                    active_details.append({
                        "symbol": symbol,
                        "state": state,
                        "order_qty": order_qty,
                        "cum_qty": cum_qty,
                        "side": side
                    })
                if log:
                    logger.info(f"⏳ 未約定(予約・待機中)の注文を認識しました: {symbol} (State: {state}, 注文数: {order_qty}, 約定済: {cum_qty})")
    return active_count, active_symbols, active_details

# --- ポジション管理 ---
@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    take_profit_price: float
    stop_loss_price: float
    hold_id: str = None
    exchange: int = 1
    last_logged_price: float = 0.0

class PortfolioManager:
    def __init__(self, config: Config, api: KabuAPI):
        self.config = config
        self.api = api
        self.positions: Dict[str, Position] = {}

    def add_position(self, symbol: str, qty: int, entry_price: float, hold_id: str = None, exchange: int = 1):
        tp = entry_price * (1 + self.config.TAKE_PROFIT_PCT)
        sl = entry_price * (1 - self.config.STOP_LOSS_PCT)
        self.positions[symbol] = Position(symbol, qty, entry_price, tp, sl, hold_id, exchange)
        mode_str = "現物" if self.config.TRADE_MODE == "CASH" else "信用"
        logger.info(f"💼 ポジション登録 [{mode_str}]: {symbol} {qty}株 (取得単価: {entry_price:,.1f}円 / 市場: {exchange})")
        logger.info(f"🎯 バリア設定 -> 利確: {tp:,.1f}円 / 損切: {sl:,.1f}円")

    async def sync_positions(self, is_startup=False):
        if is_startup:
            logger.info("🔄 持ち越しポジション（残高）を確認中...")
            
        product = 1 if self.config.TRADE_MODE == "CASH" else 2
        positions_data = await self.api.get_positions(product=product)
        
        if not positions_data:
            if is_startup:
                logger.info("保有しているポジションはありませんでした。")
            return

        found_positions = False
        for p in positions_data:
            symbol = p.get("Symbol")
            
            # 🔥 究極の修正: APIの仕様では実際の保有数は「LeavesQty」に格納されます
            leaves_qty = int(p.get("LeavesQty", 0) or 0)
            hold_qty = int(p.get("HoldQty", 0) or 0)
            qty = leaves_qty + hold_qty
            
            entry_price = float(p.get("Price", 0) or 0)
            hold_id = p.get("HoldID")
            exchange = p.get("Exchange", self.config.EXCHANGE)
            
            if qty > 0 and symbol:
                found_positions = True
                if symbol not in self.positions:
                    self.add_position(symbol, qty, entry_price, hold_id, exchange)
                    if not is_startup:
                        logger.info(f"🎉 注文の約定を確認しました！正式に監視モードに移行します: {symbol}")
                    else:
                        logger.info(f"📥 既存・新規ポジションを認識しました: {symbol} (市場: {exchange})")
                else:
                    pos = self.positions[symbol]
                    if pos.hold_id is None and hold_id is not None:
                        pos.hold_id = hold_id
                        pos.entry_price = entry_price
                        pos.take_profit_price = entry_price * (1 + self.config.TAKE_PROFIT_PCT)
                        pos.stop_loss_price = entry_price * (1 - self.config.STOP_LOSS_PCT)
                        logger.info(f"🔄 約定完了！ {symbol} の情報を最新化しました(正確な取得単価: {entry_price:,.1f}円)")
        
        # 認識できる株が1つもなかった時のログを補完
        if is_startup and not found_positions and len(positions_data) > 0:
            logger.info("保有しているポジションはありませんでした。")

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
            
            if current_price != pos.last_logged_price:
                logger.info(f"👀 {symbol} 監視中... 現在値:{current_price:,.1f}円 (利確目標:{pos.take_profit_price:,.1f}円 / 損切防衛:{pos.stop_loss_price:,.1f}円)")
                pos.last_logged_price = current_price

            if current_price >= pos.take_profit_price:
                logger.warning(f"📈 利確バリア到達！ {symbol} の決済（売り）を実行します。")
                await self.execute_exit(pos)
            elif current_price <= pos.stop_loss_price:
                logger.error(f"📉 損切バリア到達！ {symbol} の決済（売り）を実行します。")
                await self.execute_exit(pos)

    async def execute_exit(self, pos: Position):
        res = await self.api.send_order(pos.symbol, side="1", qty=pos.qty, is_close=True, hold_id=pos.hold_id, exchange=pos.exchange)
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
    
    await portfolio.sync_positions(is_startup=True)

    active_orders_count, active_order_symbols, _ = await get_active_orders(api, log=True)

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
                    
                if sig['symbol'] in active_order_symbols:
                    logger.info(f"🚫 [見送り] {sig['name']} ({sig['symbol']}) は現在「注文中(予約中)」のため追加購入しません。")
                    continue
                    
                if len(portfolio.positions) + active_orders_count >= config.MAX_POSITIONS:
                    logger.warning(f"🚫 最大保有数(保有 {len(portfolio.positions)} + 注文中 {active_orders_count} = 最大 {config.MAX_POSITIONS})に達しているため、新規エントリーを見送ります。")
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
                    active_orders_count += 1
                else:
                    logger.warning(f"⚠️ 注文送信に失敗しました。レスポンス: {res}")
        
        has_force_closed_today = False
        
        if portfolio.positions or active_orders_count > 0:
            logger.info("=== ⏱ リアルタイムインメモリ監視ループ開始 ===")
            
            sync_counter = 0 
            last_logged_active_prices = {}
            
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

                sync_counter += 1
                if sync_counter >= 3:
                    await portfolio.sync_positions(is_startup=False)
                    sync_counter = 0

                active_count, _, active_details = await get_active_orders(api, log=False)
                for detail in active_details:
                    sym = detail['symbol']
                    board = await api.get_board(sym, config.EXCHANGE)
                    if board:
                        current_price = board.get("CurrentPrice") or board.get("PreviousClose")
                        if current_price and current_price != last_logged_active_prices.get(sym):
                            action = "買い" if detail['side'] == "2" else "決済"
                            logger.info(f"⏳ {sym} 約定待ち({action})... 現在値:{current_price:,.1f}円 (注文:{detail['order_qty']}株 / 約定済:{detail['cum_qty']}株)")
                            last_logged_active_prices[sym] = current_price

                await portfolio.check_barriers()
                
                if not portfolio.positions and config.TRADE_STYLE == "day":
                    if active_count == 0:
                        await portfolio.sync_positions(is_startup=False)
                        if not portfolio.positions:
                            logger.info("📉 全てのポジションの決済が完了し、未約定の注文もありません。")
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