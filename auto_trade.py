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
# kabuステーション 自動取引エンジン (V2.1 完全突合・品質向上版)
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

# ===== V2.1: トレード実行ログ設定 =====
EXEC_LOG_PATH = "trade_execution_log.csv"
EXEC_LOG_PATH_SIM = "trade_execution_log_SIM.csv"

EXEC_LOG_HEADER = [
    "order_id", "execution_id", "order_sent_time", "fill_time", "execution_order",
    "symbol", "side", "expected_ask", "expected_bid", "actual_price", "qty",
    "spread_pct", "slippage_yen"
]

ORDER_STATUS_LOG_PATH = "order_status_log.csv"
ORDER_STATUS_LOG_PATH_SIM = "order_status_log_SIM.csv"

def ensure_exec_log_header(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(EXEC_LOG_HEADER)

def log_trade_execution(row: dict, is_sim: bool = False):
    path = EXEC_LOG_PATH_SIM if is_sim else EXEC_LOG_PATH
    ensure_exec_log_header(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=EXEC_LOG_HEADER)
        w.writerow(row)

def log_order_status(row: dict, is_sim: bool = False):
    path = ORDER_STATUS_LOG_PATH_SIM if is_sim else ORDER_STATUS_LOG_PATH
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=ORDER_STATUS_HEADER)
        if not file_exists: w.writeheader()
        w.writerow(row)

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def extract_best_bid_ask(board: dict) -> tuple[float, float]:
    """kabuステ boardレスポンスから最良気配値を取得"""
    # 🔥 Noneだけでなく、辞書型以外（エラー文字列など）が返ってきた場合のクラッシュも防ぐ
    if board is None or not isinstance(board, dict):
        return (0.0, 0.0)
    ask = board.get("Sell1", {}).get("Price") or board.get("AskPrice") or board.get("AskPrice1") or board.get("Ask") or 0.0
    bid = board.get("Buy1", {}).get("Price") or board.get("BidPrice") or board.get("BidPrice1") or board.get("Bid") or 0.0
    return (float(ask or 0.0), float(bid or 0.0))

def extract_actual_price(api_order: dict) -> float:
    """ordersレスポンスの詳細から平均約定単価（VWAP）を抽出"""
    if not api_order: return 0.0
    details = api_order.get("Details", [])
    total_val = 0.0
    total_qty = 0
    for d in details:
        if d.get("State") == 5:
            p = float(d.get("Price", 0))
            q = int(d.get("Qty", 0))
            total_val += p * q
            total_qty += q
    if total_qty > 0:
        return total_val / total_qty
    return 0.0

# V2.1: Kill Switch グローバル変数
INITIAL_EQUITY = 1_000_000
DAILY_LOSS_LIMIT = INITIAL_EQUITY * -0.02  # -2%
today_realized_pnl = 0.0
trading_halted = False

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
            "TARGET_HORIZON": "短期",         
            "TRADE_MODE": "MARGIN",
            "ACCOUNT_TYPE": 4,
            "MAX_POSITIONS": 2,
            "LOT_CALC_MODE": "FIXED", 
            "FIXED_LOT_SIZE": 100,
            "AUTO_INVEST_RATIO": 0.3,
            "ENTRY_THRESHOLD_PROB": 55.0,
            "TAKE_PROFIT_PCT": 0.05,
            "STOP_LOSS_PCT": 0.05
        }

        if not os.path.exists(self.config_file):
            self.config_data = default_config
        else:
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config_data = yaml.safe_load(f)
                logger.info(f"⚙️ 設定ファイル '{self.config_file}' を読み込みました。")
            except Exception as e:
                logger.error(f"❌ 設定ファイルの読み込みに失敗しました。デフォルト設定を使用します: {e}")
                self.config_data = default_config

        self.IS_PRODUCTION = self.config_data.get("IS_PRODUCTION", default_config["IS_PRODUCTION"])
        self.PORT = 18080 if self.IS_PRODUCTION else 18081
        self.API_URL = f"http://localhost:{self.PORT}/kabusapi"
        
        self.API_PASSWORD = str(self.config_data.get("API_PASSWORD", default_config["API_PASSWORD"]))
        self.TRADE_PASSWORD = str(self.config_data.get("TRADE_PASSWORD", default_config["TRADE_PASSWORD"]))
        self.EXCHANGE = int(self.config_data.get("EXCHANGE", default_config["EXCHANGE"]))
        
        self.TRADE_STYLE = self.config_data.get("TRADE_STYLE", default_config["TRADE_STYLE"])
        self.TARGET_HORIZON = str(self.config_data.get("TARGET_HORIZON", default_config["TARGET_HORIZON"]))

        self.TRADE_MODE = self.config_data.get("TRADE_MODE", default_config["TRADE_MODE"])
        self.ACCOUNT_TYPE = int(self.config_data.get("ACCOUNT_TYPE", default_config["ACCOUNT_TYPE"]))
        
        self.MAX_POSITIONS = self.config_data.get("MAX_POSITIONS", default_config["MAX_POSITIONS"])
        self.LOT_CALC_MODE = self.config_data.get("LOT_CALC_MODE", default_config["LOT_CALC_MODE"])
        self.FIXED_LOT_SIZE = self.config_data.get("FIXED_LOT_SIZE", default_config["FIXED_LOT_SIZE"])
        self.AUTO_INVEST_RATIO = float(self.config_data.get("AUTO_INVEST_RATIO", default_config["AUTO_INVEST_RATIO"]))
        
        self.ENTRY_THRESHOLD_PROB = float(self.config_data.get("ENTRY_THRESHOLD_PROB", default_config["ENTRY_THRESHOLD_PROB"]))
        self.TAKE_PROFIT_PCT = float(self.config_data.get("TAKE_PROFIT_PCT", default_config["TAKE_PROFIT_PCT"]))
        self.STOP_LOSS_PCT = float(self.config_data.get("STOP_LOSS_PCT", default_config["STOP_LOSS_PCT"]))
        
        # 🔥 追加: ATR連動エグジット用の係数 (k1, k2)
        self.ATR_K1 = float(self.config_data.get("ATR_K1", 2.0))
        self.ATR_K2 = float(self.config_data.get("ATR_K2", 3.0))

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
            self.session = None

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
        target_ex = 1 if int(exchange) == 9 else int(exchange)
        return await self._request("GET", f"board/{symbol}@{target_ex}")

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
        else:
            cash_margin = 3 if is_close else 2
            margin_trade_type = 1 
            deliv_type = 0

        target_exchange = exchange if exchange is not None else self.config.EXCHANGE
        
        if is_close and target_exchange == 9:
            target_exchange = 1

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
            "FundType": "  ", 
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
                    logger.info(f"⏳ 未約定(予約・待機中)の注文を認識しました: {symbol}")
    return active_count, active_symbols, active_details, orders

@dataclass
class Position:
    symbol: str; qty: int; entry_price: float; stop_loss_price: float; highest_price: float; 
    hold_id: str; exchange: int; is_closing: bool = False
    atr: float = 0.0 # 🔥 追加: ポジションにATR情報を保持

class PortfolioManager:
    def __init__(self, config: Config, api: KabuAPI):
        self.config, self.api = config, api
        self.positions, self.seen_exec_ids, self.pending_orders = {}, set(), {}
        self.active_signals: Dict[str, float] = {} # 🔥 追加: 読み込んだATRを保持する辞書
        self.is_sim = not self.config.IS_PRODUCTION # 🔥 追加: 渡し忘れ防止のためSIM判定を保持

    def _find_best_pending(self, symbol: str, side: str):
        """時間差が最小のpending（期待値ログ）を探して取り出す"""
        pending_list = self.pending_orders.get(symbol, [])
        side_pendings = [p for p in pending_list if p.get("side", "BUY") == side]
        if not side_pendings:
            return None
        
        now = time.time()
        # 注文送信時刻からの経過時間が最も小さいものを優先
        best_p = min(side_pendings, key=lambda p: abs(now - p["time_added"]))
        
        self.pending_orders[symbol] = [p for p in pending_list if p != best_p]
        if not self.pending_orders[symbol]:
            del self.pending_orders[symbol]
        return best_p

    def add_position(self, symbol: str, qty: int, entry_price: float, exchange: int, hold_id: str):
        # 🔥 変更: ATR連動の初期損切ライン(k1)を計算
        atr = self.active_signals.get(symbol, 0.0)
        stop_pct = self.config.ATR_K1 * atr if atr > 0 else self.config.STOP_LOSS_PCT
        sl = entry_price * (1 - stop_pct)
        
        self.positions[symbol] = Position(symbol, qty, entry_price, sl, entry_price, hold_id, exchange, False, atr)
        logger.info(f"💼 ポジション登録: {symbol} {qty}株 (単価: {entry_price:,.1f}円 / 損切幅: {stop_pct*100:.1f}%)")

    async def sync_positions(self, is_startup=False, all_orders=None):
        if is_startup:
            logger.info("🔄 持ち越しポジション（残高）を確認中...")
            
        product = 1 if self.config.TRADE_MODE == "CASH" else 2
        positions_data = await self.api.get_positions(product=product)
        if positions_data is None: return

        # 現在の保有建玉IDのセットを作成
        current_hold_ids = {str(p.get("HoldID", p.get("ExecutionID", ""))): p for p in positions_data}
        
        # 🔥 1. SELL（決済）の検知：持っていたはずのポジションがAPIから消えたら約定とみなす
        for symbol, pos in list(self.positions.items()):
            if pos.hold_id and pos.hold_id not in current_hold_ids:
                logger.info(f"💨 決済完了(ポジション消滅)を確認: {symbol}")
                pending = self._find_best_pending(symbol, "SELL")
                
                # 正確な約定単価を取るため orders から探す
                actual_price = 0.0
                if all_orders and pending:
                    for o in all_orders:
                        if str(o.get("ID")) == pending["order_id"]:
                            actual_price = extract_actual_price(o)
                            break
                            
                # ordersから取れなければ期待Bidで仮埋め
                if actual_price == 0.0 and pending:
                    actual_price = pending["expected_bid"]
                    
                if pending:
                    expected_bid = pending["expected_bid"]
                    # SELL側のスリッページは「期待Bidより安く売らされたらプラス（不利）」
                    slippage_yen = expected_bid - actual_price if expected_bid > 0 else 0.0
                    log_trade_execution({
                        "order_id": pending["order_id"],
                        "execution_id": pos.hold_id,
                        "order_sent_time": pending["order_sent_time"],
                        "fill_time": now_str(),
                        "execution_order": pending["execution_order"],
                        "symbol": symbol,
                        "side": "SELL",
                        "expected_ask": pending["expected_ask"],
                        "expected_bid": expected_bid,
                        "actual_price": float(actual_price),
                        "qty": pos.qty,
                        "spread_pct": pending["spread_pct"],
                        "slippage_yen": slippage_yen
                    }, is_sim=self.is_sim)
                    logger.info(f"📝 SELLログ記録: {symbol} 予想Bid:{expected_bid}円 -> 実約定:{actual_price}円 / Slip:{slippage_yen}円")
                
                # 管理から削除
                del self.positions[symbol]

        # 🔥 2. BUY（新規）の検知
        found_positions = False
        for p in positions_data:
            symbol = p.get("Symbol")
            qty = int(p.get("LeavesQty", 0) or 0) + int(p.get("HoldQty", 0) or 0)
            entry_price = float(p.get("Price", 0) or 0)
            hold_id = str(p.get("HoldID", p.get("ExecutionID", "")))
            exchange = int(p.get("Exchange", 1))
            if exchange == 9: exchange = 1
            
            if qty > 0 and symbol:
                found_positions = True
                if symbol not in self.positions:
                    self.add_position(symbol, qty, entry_price, exchange, hold_id)
                    
                    if hold_id and hold_id not in self.seen_exec_ids:
                        self.seen_exec_ids.add(hold_id)
                        pending = self._find_best_pending(symbol, "BUY")
                        
                        if pending:
                            expected_ask = pending["expected_ask"]
                            # BUY側のスリッページは「期待Askより高く買わされたらプラス（不利）」
                            slippage_yen = entry_price - expected_ask if expected_ask > 0 else 0.0
                            log_trade_execution({
                                "order_id": pending["order_id"],
                                "execution_id": hold_id,
                                "order_sent_time": pending["order_sent_time"],
                                "fill_time": now_str(),
                                "execution_order": pending["execution_order"],
                                "symbol": symbol,
                                "side": "BUY",
                                "expected_ask": expected_ask,
                                "expected_bid": pending["expected_bid"],
                                "actual_price": float(entry_price),
                                "qty": int(qty),
                                "spread_pct": pending["spread_pct"],
                                "slippage_yen": slippage_yen
                            }, is_sim=self.is_sim)
                            logger.info(f"📝 BUYログ記録: {symbol} 予想Ask:{expected_ask}円 -> 実約定:{entry_price}円 / Slip:{slippage_yen}円")
                        else:
                            # 手動発注などの場合
                            log_trade_execution({
                                "order_id": "", "execution_id": hold_id, "order_sent_time": "",
                                "fill_time": now_str(), "execution_order": 0, "symbol": symbol,
                                "side": "BUY", "expected_ask": 0.0, "expected_bid": 0.0,
                                "actual_price": float(entry_price), "qty": int(qty),
                                "spread_pct": 0.0, "slippage_yen": 0.0
                            }, is_sim=self.is_sim)
                        
                    if not is_startup:
                        logger.info(f"🎉 注文の約定を確認しました！監視モードに移行します: {symbol} (建玉ID: {hold_id})")
                    else:
                        logger.info(f"📥 既存ポジションを認識しました: {symbol} (建玉ID: {hold_id})")
                else:
                    pos = self.positions[symbol]
                    if pos.hold_id is None and hold_id is not None:
                        pos.hold_id = hold_id
                        pos.entry_price = entry_price
                        pos.highest_price = entry_price
                        pos.stop_loss_price = entry_price * (1 - self.config.STOP_LOSS_PCT)
                        logger.info(f"🔄 約定完了！ {symbol} の情報を最新化しました(建玉ID: {hold_id})")
        
        if is_startup and not found_positions and len(positions_data) > 0:
            logger.info("保有しているポジションはありませんでした。")

    async def cleanup_pendings(self, all_orders):
        """TTL超過の保留ログを状態判定して破棄する"""
        current_time = time.time()
        orders_by_id = {str(o.get("ID")): o for o in (all_orders or [])}
        
        for sym in list(self.pending_orders.keys()):
            active_pendings = []
            for p in self.pending_orders[sym]:
                order_id = p["order_id"]
                api_order = orders_by_id.get(order_id)
                
                if current_time - p["time_added"] > 600: # 10分経過
                    status = "TIMEOUT"
                    reason = "TTL (10min) expired"
                    
                    if api_order:
                        state = api_order.get("State")
                        cum_qty = int(api_order.get("CumQty", 0))
                        if state == 5:
                            if cum_qty == 0:
                                status = "CANCELED/REJECTED"
                                reason = "Order completed with 0 qty"
                            else:
                                status = "PARTIAL_FILLED"
                                reason = f"Completed with {cum_qty} qty"
                        else:
                            status = f"API_STATE_{state}"
                            
                    log_order_status({
                        "order_id": order_id, "symbol": sym, "side": p.get("side", "BUY"),
                        "expected_ask": p["expected_ask"], "expected_bid": p["expected_bid"],
                        "order_sent_time": p["order_sent_time"],
                        "status": status, "reason": reason
                    }, is_sim=self.is_sim)
                    logger.info(f"🗑️ ログ待機破棄: {sym} ({status})")
                    continue
                
                active_pendings.append(p)
                
            self.pending_orders[sym] = active_pendings
            if not self.pending_orders[sym]:
                del self.pending_orders[sym]

    async def check_barriers(self):
        """トレール決済・損切監視"""
        for sym, pos in list(self.positions.items()):
            if pos.is_closing: continue

            board = await self.api.get_board(sym, pos.exchange)
            if not board: continue
            
            current_price = board.get("CurrentPrice") or board.get("PreviousClose")
            
            if not self.config.IS_PRODUCTION and current_price is None:
                import random
                fluctuation = pos.entry_price * random.uniform(-0.1, 0.1)
                current_price = pos.entry_price + fluctuation

            if current_price is None: continue

            if current_price > pos.highest_price:
                pos.highest_price = current_price
                if current_price > pos.entry_price:
                    # 🔥 変更: ATR連動のトレール幅(k2)を計算
                    trail_pct = self.config.ATR_K2 * pos.atr if pos.atr > 0 else self.config.STOP_LOSS_PCT
                    new_sl = current_price * (1 - trail_pct)
                    if new_sl > pos.stop_loss_price:
                        pos.stop_loss_price = new_sl
                        logger.info(f"✨ トレール発動！ {sym} が最高値を更新。損切ラインを {new_sl:,.1f}円 に引き上げました。")
            
            if current_price <= pos.stop_loss_price:
                if current_price > pos.entry_price:
                    logger.warning(f"📈 トレール利確！ {sym} の決済（売り）を実行します。")
                else:
                    logger.error(f"📉 損切バリア到達！ {sym} の決済（売り）を実行します。")
                await self.execute_exit(pos)

    async def execute_exit(self, pos: Position):
        pos.is_closing = True # 二重決済防止
        board = await self.api.get_board(pos.symbol, self.config.EXCHANGE)
        ask1, bid1 = extract_best_bid_ask(board)
        exit_est = bid1 if bid1 > 0 else ((board.get("CurrentPrice") if board else None) or pos.entry_price)

        realized = (exit_est - pos.entry_price) * pos.qty

        global today_realized_pnl, trading_halted, DAILY_LOSS_LIMIT
        today_realized_pnl += realized
        if today_realized_pnl <= DAILY_LOSS_LIMIT:
            trading_halted = True
            logger.error(f"🛑 KillSwitch発動: 本日確定損益={today_realized_pnl:,.0f}円（閾値={DAILY_LOSS_LIMIT:,.0f}円）")

        target_exchange = 1
        res = await self.api.send_order(pos.symbol, side="1", qty=pos.qty, is_close=True, hold_id=pos.hold_id, exchange=target_exchange)
        if res and res.get("Result") == 0:
            order_id = str(res.get("OrderId", ""))
            logger.info(f"✅ 決済注文受付成功: {pos.symbol} (OrderID: {order_id})")

            if not self.config.IS_PRODUCTION and str(order_id).startswith('SIM_'):
                log_order_status({
                    'order_id': order_id,
                    'symbol': pos.symbol,
                    'side': 'SELL',
                    'expected_ask': ask1,
                    'expected_bid': bid1,
                    'order_sent_time': now_str(),
                    'status': 'SIM_SENT',
                    'reason': 'Simulated close order (no execution)'
                }, is_sim=self.is_sim)
                return

            spread_pct = ((ask1 - bid1) / bid1) if bid1 > 0 else 0.0
            self.pending_orders.setdefault(pos.symbol, []).append({
                'order_id': order_id,
                'order_sent_time': now_str(),
                'execution_order': 0,
                'expected_ask': ask1,
                'expected_bid': bid1,
                'spread_pct': spread_pct,
                'time_added': time.time(),
                'side': 'SELL'
            })
        else:
            pos.is_closing = False
            logger.error(f"❌ 決済注文失敗: {res}")

def get_ticker_mapping() -> dict:
    mapping = {}
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r", encoding="utf-8") as f:
            for line in f:
                if ',' in line:
                    name, tk = line.split(',', 1)
                    mapping[name.strip()] = tk.replace(".T", "").strip()
    return mapping

def load_ai_signals(config: Config) -> List[dict]:
    signals = []
    if not os.path.exists('recommendations.csv'): return signals
    mapping = get_ticker_mapping()
    
    with open('recommendations.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        has_net = "Net_Score(%)" in (reader.fieldnames or [])

        for row in reader:
            name = row.get('銘柄名', '')
            code = row.get("銘柄コード") or mapping.get(name)
            if not code: continue
            code = str(code).replace(".T", "").strip()

            atr = 0.0
            try: atr = float(row.get("ATR_Prev_Ratio", 0.0))
            except ValueError: pass

            if has_net:
                try: net_pct = float(row.get("Net_Score(%)", 0.0))
                except ValueError: net_pct = 0.0
                if net_pct <= 0: continue
                
                try: prob = float(row.get("メタ確信度(%)", row.get("メタ確信度", 50.0)))
                except ValueError: prob = 50.0
                
                prob = max(0.0, min(prob, 100.0))
                signals.append({"name": name, "symbol": code, "net_pct": net_pct, "prob": prob, "atr": atr})
            else:
                try: prob = float(row.get("短期スコア", 0))
                except ValueError: prob = 0.0
                
                prob = max(0.0, min(prob, 100.0))
                if prob >= config.ENTRY_THRESHOLD_PROB:
                    signals.append({'name': name, 'symbol': code, 'prob': prob, "atr": atr})
                
    if signals and "net_pct" in signals[0]:
        signals.sort(key=lambda x: x["net_pct"], reverse=True)
    else:
        signals.sort(key=lambda x: x.get('prob', 0), reverse=True)
    return signals


async def main():
    global today_realized_pnl, trading_halted, DAILY_LOSS_LIMIT
    
    config = Config()
    logger.info(f"=== 自動取引エンジン起動 (モード: {config.TRADE_STYLE.upper()}) ===")
    
    api = KabuAPI(config)
    await api.start_session()
    await api.get_token()
    if not api.token: return

    portfolio = PortfolioManager(config, api)
    await portfolio.sync_positions(is_startup=True)
    active_orders_count, active_order_symbols, _, all_orders = await get_active_orders(api, log=True)

    try:
        now = datetime.now()
        if config.TRADE_STYLE == "day" and (now.hour > 14 or (now.hour == 14 and now.minute >= 30)):
            logger.info("🕒 14:30を過ぎているため、本日の新規エントリーは見送ります（デイトレモード）。")
            signals = []
        else:
            signals = load_ai_signals(config)
            # 🔥 追加: 読み込んだシグナルのATRをPortfolioManagerに登録
            portfolio.active_signals = {sig['symbol']: sig.get('atr', 0.0) for sig in signals}

        if not signals:
            logger.info("😴 本日は新規エントリー条件を満たす銘柄はありませんでした。")
        else:
            execution_order = 0
            consecutive_board_failures = 0
            for sig in signals:
                if trading_halted:
                    logger.warning("🛑 KillSwitch: 本日は損失上限に達したため新規エントリーを停止します。")
                    break

                if sig['symbol'] in portfolio.positions: continue
                if sig['symbol'] in active_order_symbols: continue
                
                if len(portfolio.positions) + active_orders_count >= config.MAX_POSITIONS:
                    logger.warning(f"🚫 最大保有数({config.MAX_POSITIONS}銘柄)に達したため、これ以上の新規エントリーを見送ります。")
                    break

                logger.info(f"🌟 AIシグナル発火！ 銘柄: {sig['name']} ({sig['symbol']})")
                execution_order += 1
                board = await api.get_board(sig['symbol'], config.EXCHANGE)
                if not board:
                    consecutive_board_failures += 1
                    logger.warning(f"⚠️ 板情報が取得できませんでした: {sig['symbol']}（スキップ）")
                    if consecutive_board_failures >= 5:
                        logger.error("🛑 板情報取得の連続失敗が閾値に達したため、本日の新規エントリーを停止します")
                        break
                    continue

                consecutive_board_failures = 0

                ask1, bid1 = extract_best_bid_ask(board)
                if ask1 <= 0:
                    logger.warning(f"⚠️ 最良売気配(Ask)が0です: {sig['symbol']}（スキップ）")
                    continue

                spread_pct = ((ask1 - bid1) / bid1) if bid1 > 0 else 0.0
                ref_price = ask1 

                qty = 0
                if config.LOT_CALC_MODE == "FIXED":
                    qty = config.FIXED_LOT_SIZE
                elif config.LOT_CALC_MODE == "AUTO":
                    avail = (await api.get_wallet_cash()).get("StockAccountWallet", 1000000)
                    qty = (int((avail * config.AUTO_INVEST_RATIO) / config.MAX_POSITIONS / ref_price) // 100) * 100 
                elif config.LOT_CALC_MODE == "KELLY":
                    avail = (await api.get_wallet_cash()).get("StockAccountWallet", 1000000)
                    b = config.TAKE_PROFIT_PCT / config.STOP_LOSS_PCT if config.STOP_LOSS_PCT > 0 else 1.0
                    
                    prob_pct = float(sig.get('prob', 50.0))
                    prob_pct = max(0.0, min(prob_pct, 100.0))
                    p = prob_pct / 100.0
                    
                    kelly_f = p - ((1.0 - p) / b) if b > 0 else 0.0
                    invest_ratio = max(0.0, min(kelly_f / 2.0, config.AUTO_INVEST_RATIO))
                    
                    if invest_ratio > 0:
                        qty = (int((avail * invest_ratio) / ref_price) // 100) * 100 
                        logger.info(f"🧠 ケリー基準計算: 勝率={p*100:.1f}%, 投資割合={invest_ratio*100:.1f}% -> 算出ロット={qty}株")
                    else:
                        logger.warning(f"⚠️ ケリー基準がマイナスのため見送ります。(勝率: {p*100:.1f}%)")
                        qty = 0

                avail = (await api.get_wallet_cash()).get("StockAccountWallet", 1_000_000)
                max_notional = avail * 0.10
                max_qty_notional = int(max_notional // ref_price) if ref_price > 0 else 0

                stop_pct = config.STOP_LOSS_PCT if config.STOP_LOSS_PCT > 0 else 0.05
                max_risk = avail * 0.005
                max_qty_risk = int(max_risk // (ref_price * stop_pct)) if ref_price > 0 else 0

                max_qty = max(0, min(max_qty_notional, max_qty_risk))
                max_qty = (max_qty // 100) * 100
                
                qty = min(qty, max_qty)
                if qty <= 0:
                    logger.info("⛔ 1銘柄上限によりロットが0になったためスキップ")
                    continue

                order_sent_time = now_str()
                res = await api.send_order(sig['symbol'], side="2", qty=qty, is_close=False)
                
                if res and res.get("Result") == 0:
                    order_id = str(res.get("OrderId", ""))
                    logger.info(f"✅ エントリー注文送信成功 order_id={order_id}")
                    
                    if not config.IS_PRODUCTION and str(order_id).startswith('SIM_'):
                        log_order_status({
                            'order_id': order_id,
                            'symbol': sig['symbol'],
                            'side': 'BUY',
                            'expected_ask': ask1,
                            'expected_bid': bid1,
                            'order_sent_time': order_sent_time,
                            'status': 'SIM_SENT',
                            'reason': 'Simulated order (no execution)'
                        }, is_sim=not config.IS_PRODUCTION)
                    else:
                        portfolio.pending_orders.setdefault(sig['symbol'], []).append({
                            'order_id': order_id,
                            'order_sent_time': order_sent_time,
                            'execution_order': execution_order,
                            'expected_ask': ask1,
                            'expected_bid': bid1,
                            'spread_pct': spread_pct,
                            'time_added': time.time(),
                            'side': 'BUY'
                        })
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
                            if not pos.is_closing:
                                await portfolio.execute_exit(pos)
                        has_force_closed_today = True
                    if len(portfolio.positions) == 0:
                        break
                    
                if now.hour >= 15:
                    logger.info("🌙 15:00を回りました。本日の市場監視を終了します。")
                    break

                active_count, _, active_details, all_orders = await get_active_orders(api, log=False)
                
                sync_counter += 1
                if sync_counter >= 3:
                    await portfolio.sync_positions(is_startup=False, all_orders=all_orders)
                    sync_counter = 0

                await portfolio.cleanup_pendings(all_orders)

                for detail in active_details:
                    sym = detail['symbol']
                    board = await api.get_board(sym, config.EXCHANGE)
                    if board:
                        current_price = board.get("CurrentPrice") or board.get("PreviousClose")
                        if current_price and current_price != last_logged_active_prices.get(sym):
                            last_logged_active_prices[sym] = current_price

                await portfolio.check_barriers()
                
                if not portfolio.positions and config.TRADE_STYLE == "day":
                    if active_count == 0:
                        await portfolio.sync_positions(is_startup=False, all_orders=all_orders)
                        if not portfolio.positions:
                            logger.info("📉 全てのポジションの決済が完了し、未約定の注文もありません。本日の取引を終了します。")
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