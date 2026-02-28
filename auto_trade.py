import asyncio
import time
import logging
import aiohttp
import csv
import os
import yaml
from dataclasses import dataclass
from typing import Dict, List

# =========================================================
# kabuステーション 自動取引エンジン (auto_trade.py) - 統合本番仕様
# ---------------------------------------------------------
# 【特徴】
# 1. AI予測結果 (recommendations.csv) の自動読み込みとシグナル抽出
# 2. 現物/信用のシームレスな切り替え（完全攻略版APIパラメータ）
# 3. TokenBucketによるAPI制限(秒間5件)の回避
# 4. インメモリストートマシンによるリアルタイムバリア監視(利確/損切)
# 5. 設定の外部ファイル化 (settings.yml)
# 6. 【NEW】買付余力の自動取得によるロット数（購入株数）自動計算機能
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
        # デフォルトの設定値
        default_config = {
            "IS_PRODUCTION": False,
            "API_PASSWORD": "YOUR_API_PASSWORD",
            "TRADE_PASSWORD": "YOUR_TRADE_PASSWORD",
            "EXCHANGE": 1,
            "TRADE_MODE": "CASH",
            "MAX_POSITIONS": 1,
            "LOT_CALC_MODE": "AUTO",          # FIXED:固定株数, AUTO:残高から自動計算
            "FIXED_LOT_SIZE": 100,
            "AUTO_INVEST_RATIO": 0.9,         # 自動計算時の資金利用上限(90%)
            "ENTRY_THRESHOLD_PROB": 60.0,
            "TAKE_PROFIT_PCT": 0.05,
            "STOP_LOSS_PCT": 0.05
        }

        # 設定ファイルが存在しない場合はデフォルト設定でYAMLを作成する
        if not os.path.exists(self.config_file):
            try:
                # デフォルト設定を書き込む際のコメント入りテンプレート
                yaml_template = """# ==========================================
# 自動取引エンジン 設定ファイル (settings.yml)
# ==========================================

# --- 環境・認証設定 ---
# 本番環境モード。falseでシミュレーター(ポート18081)、trueで本番(ポート18080)に接続します
IS_PRODUCTION: false

# kabuステーションのAPI接続パスワード（検証環境を使う場合は検証用のもの）
API_PASSWORD: "YOUR_API_PASSWORD"

# 注文・決済時に使用する取引パスワード（証券口座のもの）
TRADE_PASSWORD: "YOUR_TRADE_PASSWORD"

# 取引所コード。1は「東証」を意味します
EXCHANGE: 1

# --- 取引・ロット設定 ---
# 取引モード。「CASH」で現物取引、「MARGIN」で信用取引（制度信用）を行います
TRADE_MODE: "CASH"

# システムが同時に保有する最大銘柄数
MAX_POSITIONS: 1

# ロット数の計算モード。「FIXED」で固定株数、「AUTO」で口座残高から自動計算します
LOT_CALC_MODE: "AUTO"

# 「FIXED」モード時の1回のエントリーで購入する株数（例：100なら1単元）
FIXED_LOT_SIZE: 100

# 「AUTO」モード時、口座の買付余力（残高）に対する最大投資割合（例: 0.9 = 余力の90%を上限とする）
AUTO_INVEST_RATIO: 0.9

# --- ロジック・バリア設定 ---
# 買いシグナルを発火する「明日の上昇確率」のしきい値(%)
ENTRY_THRESHOLD_PROB: 60.0

# 利確（テイクプロフィット）のライン。0.05は取得単価の「+5%」で自動決済します
TAKE_PROFIT_PCT: 0.05

# 損切（ストップロス）のライン。0.05は取得単価の「-5%」で自動決済します
STOP_LOSS_PCT: 0.05
"""
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    f.write(yaml_template)
                logger.info(f"⚙️ 設定ファイル '{self.config_file}' を新規作成しました。")
            except Exception as e:
                logger.warning(f"⚠️ 設定ファイルの作成に失敗しました: {e}")
            config_data = default_config
        else:
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                logger.info(f"⚙️ 設定ファイル '{self.config_file}' を読み込みました。")
            except Exception as e:
                logger.error(f"❌ 設定ファイルの読み込みに失敗しました。デフォルト設定を使用します: {e}")
                config_data = default_config

        # プロパティの割り当て
        self.IS_PRODUCTION = config_data.get("IS_PRODUCTION", default_config["IS_PRODUCTION"])
        self.PORT = 18080 if self.IS_PRODUCTION else 18081
        self.API_URL = f"http://localhost:{self.PORT}/kabusapi"
        
        self.API_PASSWORD = config_data.get("API_PASSWORD", default_config["API_PASSWORD"])
        self.TRADE_PASSWORD = config_data.get("TRADE_PASSWORD", default_config["TRADE_PASSWORD"])
        self.EXCHANGE = config_data.get("EXCHANGE", default_config["EXCHANGE"])
        self.TRADE_MODE = config_data.get("TRADE_MODE", default_config["TRADE_MODE"])
        
        self.MAX_POSITIONS = config_data.get("MAX_POSITIONS", default_config["MAX_POSITIONS"])
        self.LOT_CALC_MODE = config_data.get("LOT_CALC_MODE", default_config["LOT_CALC_MODE"])
        self.FIXED_LOT_SIZE = config_data.get("FIXED_LOT_SIZE", default_config["FIXED_LOT_SIZE"])
        self.AUTO_INVEST_RATIO = config_data.get("AUTO_INVEST_RATIO", default_config["AUTO_INVEST_RATIO"])
        
        self.ENTRY_THRESHOLD_PROB = config_data.get("ENTRY_THRESHOLD_PROB", default_config["ENTRY_THRESHOLD_PROB"])
        self.TAKE_PROFIT_PCT = config_data.get("TAKE_PROFIT_PCT", default_config["TAKE_PROFIT_PCT"])
        self.STOP_LOSS_PCT = config_data.get("STOP_LOSS_PCT", default_config["STOP_LOSS_PCT"])

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

    # 💡【NEW】口座の残高（余力）を取得するAPIメソッド
    async def get_wallet_cash(self):
        """現物買付余力照会"""
        return await self._request("GET", "wallet/cash")

    async def get_wallet_margin(self, symbol: str, exchange: int):
        """信用新規建余力照会（銘柄ごとに保証金から計算）"""
        return await self._request("GET", f"wallet/margin/{symbol}@{exchange}")

    async def send_order(self, symbol: str, side: str, qty: int, price: float = 0, is_close: bool = False, hold_id: str = None):
        if self.config.TRADE_MODE == "CASH":
            cash_margin = 1
            margin_trade_type = 1 
            deliv_type = 2 if side == "2" else 0
            fund_type = "02" if side == "2" else "  "
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
            "AccountType": 4,
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
        logger.info(f"💼 ポジション追加 [{mode_str}]: {symbol} {qty}株 (取得単価: {entry_price:,.1f}円)")
        logger.info(f"🎯 バリア設定 -> 利確: {tp:,.1f}円 / 損切: {sl:,.1f}円")

    async def check_barriers(self):
        for symbol, pos in list(self.positions.items()):
            board = await self.api.get_board(symbol, self.config.EXCHANGE)
            if not board: continue
            
            current_price = board.get("CurrentPrice") or board.get("PreviousClose")
            
            # 休日テスト用ダミー
            if current_price is None:
                import random
                fluctuation = pos.entry_price * random.uniform(-0.1, 0.1)
                current_price = pos.entry_price + fluctuation

            if current_price is None: continue

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
        "任天堂": "7974.T", "オリックス": "8591.T", "ANA": "9202.T", "三井物産": "8031.T"
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
        for row in reader:
            prob = float(row.get('明日の上昇確率', 0))
            if prob >= config.ENTRY_THRESHOLD_PROB:
                name = row.get('銘柄名', '')
                code = mapping.get(name)
                if code: signals.append({'name': name, 'symbol': code, 'prob': prob})
    signals = sorted(signals, key=lambda x: x['prob'], reverse=True)
    return signals[:config.MAX_POSITIONS]

# --- メインループ ---
async def main():
    config = Config()
    logger.info(f"=== 自動取引エンジン起動 ({config.LOT_CALC_MODE}ロットモード) ===")
    
    api = KabuAPI(config)
    await api.start_session()
    await api.get_token()

    if not api.token: return

    portfolio = PortfolioManager(config, api)

    try:
        signals = load_ai_signals(config)

        if not signals:
            logger.info("😴 本日はエントリー条件を満たす銘柄はありませんでした。")
        else:
            for sig in signals:
                logger.info(f"🌟 AIシグナル発火！ 銘柄: {sig['name']} ({sig['symbol']}) - 上昇確率: {sig['prob']}%")
                
                board = await api.get_board(sig['symbol'], config.EXCHANGE)
                current_price = board.get("CurrentPrice") or board.get("PreviousClose") if board else None
                
                if not current_price:
                    current_price = 2000.0
                    logger.warning("⚠️ 価格が取得できないためダミー価格(2000円)を適用します。")

                # 💡【NEW】口座の残高からロット（株数）を自動計算するロジック
                qty = 0
                if config.LOT_CALC_MODE == "FIXED":
                    qty = config.FIXED_LOT_SIZE
                elif config.LOT_CALC_MODE == "AUTO":
                    available_cash = 0
                    # 現物か信用かで、照会するAPIを切り替える
                    if config.TRADE_MODE == "CASH":
                        wallet = await api.get_wallet_cash()
                        if wallet: available_cash = wallet.get("StockAccountWallet", 0)
                    else:
                        wallet = await api.get_wallet_margin(sig['symbol'], config.EXCHANGE)
                        if wallet: available_cash = wallet.get("MarginAccountWallet", 0)
                    
                    # 休日のシミュレーター等で余力が取れない場合のフェイルセーフ
                    if not available_cash:
                        logger.warning("⚠️ 買付余力が取得できないため、仮想余力(1,000,000円)として計算します。")
                        available_cash = 1000000

                    # 1銘柄あたりの割り当て予算 = (総余力 * 投資割合(デフォルト90%)) / 最大保有銘柄数
                    budget_per_trade = (available_cash * config.AUTO_INVEST_RATIO) / config.MAX_POSITIONS
                    max_shares = int(budget_per_trade / current_price)
                    qty = (max_shares // 100) * 100 # 100株単位で切り捨て

                    logger.info(f"💰 ロット自動計算: 余力={available_cash:,.0f}円 -> 割当予算={budget_per_trade:,.0f}円 -> 算出ロット={qty}株")

                # 資金不足で100株も買えない場合はエントリーをスキップ
                if qty == 0:
                    logger.warning(f"⚠️ 資金不足（算出ロット0株）のため、{sig['symbol']} のエントリーを見送ります。")
                    continue

                res = await api.send_order(sig['symbol'], side="2", qty=qty, is_close=False)
                
                if res and res.get("Result") == 0:
                    logger.info(f"✅ エントリー注文送信成功")
                    portfolio.add_position(sig['symbol'], qty, current_price, None)
                else:
                    logger.warning(f"⚠️ 注文送信に失敗しました。レスポンス: {res}")
        
        if portfolio.positions:
            logger.info("=== ⏱ リアルタイムインメモリ監視ループ開始 ===")
            while True:
                await portfolio.check_barriers()
                if not portfolio.positions:
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