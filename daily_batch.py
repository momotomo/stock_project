import csv
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from settings_loader import as_bool, as_float, as_int, load_settings

# =========================================================
# AI予測バッチ処理 (daily_batch.py) - V2.1 完全実運用対応版
# =========================================================

warnings.simplefilter("ignore", ResourceWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_FEE = 0.001
SLIPPAGE_FACTOR = 0.05
TIME_LAG_PENALTY = 0.001

RECOMMENDATION_COLUMNS = [
    "銘柄名",
    "銘柄コード",
    "今日の終値",
    "Net_Score(%)",
    "予測リターン(%)",
    "推定往復コスト(%)",
    "メタ確信度",
    "短期スコア",
    "おすすめ理由",
]

HEALTH_LOG_COLUMNS = [
    "run_at",
    "breaker_enabled",
    "breaker",
    "reason",
    "breaker_ticker",
    "ma_days",
    "ret_threshold",
    "cond_close_lt_ma",
    "cond_ret_lt_threshold",
    "topix_close",
    "topix_ma",
    "topix_ret1",
    "note",
]

BREAKER_EVENT_LOG_COLUMNS = [
    "timestamp",
    "symbol",
    "side_candidate",
    "intended_qty",
    "breaker_name",
    "breaker_reason",
    "market_phase",
    "price_reference",
    "volatility_reference",
    "spread_reference",
    "action_taken",
]

SIMULATED_ORDER_LOG_COLUMNS = [
    "timestamp",
    "symbol",
    "side",
    "qty",
    "order_type",
    "intended_price",
    "signal_score",
    "model_decision",
    "blocked_by_breaker",
    "blocker_reason",
    "would_open_or_close",
]

BREAKER_EVENT_LOG_PATH = "breaker_event_log.csv"
SIMULATED_ORDER_LOG_PATH = "simulated_order_log.csv"


def cost_hat_roundtrip(atr_prev_ratio: float) -> float:
    return (BASE_FEE * 2.0) + (atr_prev_ratio * SLIPPAGE_FACTOR * 2.0) + TIME_LAG_PENALTY


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def write_empty_recommendations(path: str) -> None:
    ensure_parent_dir(path)
    pd.DataFrame(columns=RECOMMENDATION_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def append_note(current: str, extra: str) -> str:
    if not extra:
        return current
    if not current:
        return extra
    return f"{current}; {extra}"


def normalize_observation_row(row: Dict[str, object], columns) -> Dict[str, object]:
    return {column: row.get(column, "") for column in columns}


def ensure_observation_log_schema(path: str, columns) -> None:
    ensure_parent_dir(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
        return

    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if fieldnames == list(columns):
            return
        existing_rows = [normalize_observation_row(row, columns) for row in reader]

    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(existing_rows)


def append_observation_rows(path: str, columns, rows) -> None:
    if not rows:
        return
    ensure_observation_log_schema(path, columns)
    with open(path, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        for row in rows:
            writer.writerow(normalize_observation_row(row, columns))


def write_health_log(path: str, row: Dict[str, object]) -> None:
    ensure_parent_dir(path)
    ensure_health_log_schema(path)
    normalized = normalize_health_log_row(row)
    with open(path, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEALTH_LOG_COLUMNS)
        writer.writerow(normalized)


def normalize_health_log_row(row: Dict[str, object]) -> Dict[str, object]:
    return {column: row.get(column, "") for column in HEALTH_LOG_COLUMNS}


def ensure_health_log_schema(path: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEALTH_LOG_COLUMNS)
            writer.writeheader()
        return

    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if fieldnames == HEALTH_LOG_COLUMNS:
            return
        existing_rows = [normalize_health_log_row(row) for row in reader]

    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEALTH_LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(existing_rows)


def get_tickers() -> Dict[str, str]:
    tickers = {}
    if not os.path.exists("tickers.txt"):
        return tickers

    with open("tickers.txt", "r", encoding="utf-8") as handle:
        for line in handle:
            if "," not in line:
                continue
            name, ticker = line.split(",", 1)
            tickers[name.strip()] = ticker.strip()
    return tickers


def get_macro_data() -> pd.DataFrame:
    macro_tickers = {"USDJPY=X": "USDJPY_Ret", "^GSPC": "SP500_Ret", "1306.T": "TOPIX_Ret"}
    try:
        macro_df = yf.download(list(macro_tickers.keys()), period="5y", progress=False)
        macro_close = macro_df.xs("Close", level=0, axis=1) if isinstance(macro_df.columns, pd.MultiIndex) else macro_df["Close"]
        macro_ret = np.log(macro_close / macro_close.shift(1)).rename(columns=macro_tickers)
        macro_ret.index = pd.to_datetime(macro_ret.index).map(lambda x: x.replace(tzinfo=None).normalize())
        return macro_ret
    except Exception as exc:
        logger.error(f"マクロデータの取得エラー: {exc}")
        return pd.DataFrame()


def calc_fractional_diff(series, d=0.5, window=20):
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights)[::-1]
    return series.rolling(window).apply(lambda x: np.dot(weights, x), raw=True)


def get_stock_features(ticker, macro_returns):
    try:
        data = yf.download(ticker, period="5y", progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.loc[:, ~data.columns.duplicated()].copy()
        data.index = pd.to_datetime(data.index).map(lambda x: x.replace(tzinfo=None).normalize())

        data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
        data["Frac_Diff_0.5"] = calc_fractional_diff(data["Close"], d=0.5, window=20)

        tr = pd.concat(
            [
                data["High"] - data["Low"],
                (data["High"] - data["Close"].shift()).abs(),
                (data["Low"] - data["Close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        data["ATR"] = tr.rolling(14).mean() / data["Close"]

        data["Disparity_5"] = (data["Close"] / data["Close"].rolling(5).mean()) - 1
        data["Disparity_25"] = (data["Close"] / data["Close"].rolling(25).mean()) - 1
        data["Log_Return_Norm"] = data["Log_Return"] / (data["ATR"] + 1e-9)
        data["Disparity_5_Norm"] = data["Disparity_5"] / (data["ATR"] + 1e-9)
        data["Disparity_25_Norm"] = data["Disparity_25"] / (data["ATR"] + 1e-9)
        data["SMA_Cross"] = (data["Close"].rolling(5).mean() / data["Close"].rolling(25).mean()) - 1

        delta = data["Close"].diff()
        ema_up = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        ema_down = (-1 * delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        data["RSI_14"] = 100 - (100 / (1 + (ema_up / (ema_down + 1e-9))))

        std_20 = data["Close"].rolling(20).std()
        ma_20 = data["Close"].rolling(20).mean()
        data["BB_PctB"] = (data["Close"] - (ma_20 - 2 * std_20)) / (4 * std_20 + 1e-9)
        data["BB_Bandwidth"] = (4 * std_20) / (data["Close"].rolling(25).mean() + 1e-9)
        data["MACD_Norm"] = (data["Close"].ewm(span=12).mean() - data["Close"].ewm(span=26).mean()) / data["Close"]
        data["OBV_Ret"] = (np.sign(data["Close"].diff()) * data["Volume"]).fillna(0).cumsum().pct_change()

        ema_20 = data["Close"].ewm(span=20, adjust=False).mean()
        data["EMA_Diff_Ratio"] = (data["Close"] - data["Open"]) / (ema_20 + 1e-9)

        price_slope = data["Close"].diff(5)
        macd_slope = data["MACD_Norm"].diff(5)
        data["MACD_Div"] = np.where(np.sign(price_slope) != np.sign(macd_slope), 1, 0) * macd_slope

        for col in ["Log_Return_Norm", "RSI_14", "MACD_Norm"]:
            data[f"{col}_Lag1"] = data[col].shift(1)
            data[f"{col}_Lag2"] = data[col].shift(2)

        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        if not macro_returns.empty:
            data = data.join(macro_returns)
        for col in ["USDJPY_Ret", "SP500_Ret", "TOPIX_Ret"]:
            if col not in data.columns:
                data[col] = 0.0
        data[["USDJPY_Ret", "SP500_Ret", "TOPIX_Ret"]] = data[["USDJPY_Ret", "SP500_Ret", "TOPIX_Ret"]].ffill().fillna(0)

        data["Excess_Return"] = data["Log_Return"] - data["TOPIX_Ret"]
        data["Excess_Return_20d"] = data["Excess_Return"].rolling(20).sum()
        return data
    except Exception as exc:
        logger.error(f"銘柄 {ticker} のデータ取得エラー: {exc}")
        return None


@dataclass
class BatchConfig:
    config_data: Dict[str, object]
    BREAKER_ENABLED: bool
    BREAKER_TICKER: str
    BREAKER_RET_THRESHOLD: float
    BREAKER_MA_DAYS: int
    HEALTH_LOG_PATH: str
    RECO_CSV_PATH: str

    @classmethod
    def load(cls) -> "BatchConfig":
        defaults = {
            "IS_PRODUCTION": False,
            "API_PASSWORD": "",
            "API_PASSWORD_SIM": "YOUR_SIM_API_PASSWORD",
            "API_PASSWORD_PROD": "YOUR_PROD_API_PASSWORD",
            "TRADE_PASSWORD": "YOUR_TRADE_PASSWORD",
            "BREAKER_ENABLED": True,
            "BREAKER_TICKER": "1306.T",
            "BREAKER_RET_THRESHOLD": -0.015,
            "BREAKER_MA_DAYS": 5,
            "HEALTH_LOG_PATH": "daily_health_log.csv",
            "RECO_CSV_PATH": "recommendations.csv",
        }
        config_data = load_settings(defaults, logger=logger)
        return cls(
            config_data=config_data,
            BREAKER_ENABLED=as_bool(config_data.get("BREAKER_ENABLED"), defaults["BREAKER_ENABLED"]),
            BREAKER_TICKER=str(config_data.get("BREAKER_TICKER", defaults["BREAKER_TICKER"])),
            BREAKER_RET_THRESHOLD=as_float(
                config_data.get("BREAKER_RET_THRESHOLD"),
                defaults["BREAKER_RET_THRESHOLD"],
            ),
            BREAKER_MA_DAYS=max(1, as_int(config_data.get("BREAKER_MA_DAYS"), defaults["BREAKER_MA_DAYS"])),
            HEALTH_LOG_PATH=str(config_data.get("HEALTH_LOG_PATH", defaults["HEALTH_LOG_PATH"])),
            RECO_CSV_PATH=str(config_data.get("RECO_CSV_PATH", defaults["RECO_CSV_PATH"])),
        )


@dataclass
class MarketBreakerResult:
    breaker_enabled: bool
    breaker: bool
    reason: str
    ticker: str
    ma_days: int
    ret_threshold: float
    cond_close_lt_ma: bool = False
    cond_ret_lt_threshold: bool = False
    close: Optional[float] = None
    ma: Optional[float] = None
    ret1: Optional[float] = None
    note: str = ""


def build_breaker_reason(cond_close_lt_ma: bool, cond_ret_lt_threshold: bool) -> str:
    reasons = []
    if cond_close_lt_ma:
        reasons.append("close_below_ma")
    if cond_ret_lt_threshold:
        reasons.append("ret_below_threshold")
    return "|".join(reasons) if reasons else "OK"


def check_market_breaker(config: BatchConfig) -> MarketBreakerResult:
    logger.info("🔍 相場環境(地合い)の悪化をチェックしています...")
    result = MarketBreakerResult(
        breaker_enabled=config.BREAKER_ENABLED,
        breaker=False,
        reason="OK",
        ticker=config.BREAKER_TICKER,
        ma_days=config.BREAKER_MA_DAYS,
        ret_threshold=config.BREAKER_RET_THRESHOLD,
    )

    try:
        lookback_days = max(60, config.BREAKER_MA_DAYS + 5)
        df = yf.download(config.BREAKER_TICKER, period=f"{lookback_days}d", progress=False)
        if df.empty:
            result.note = "breaker data unavailable"
            return result

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = float(df["Close"].iloc[-1])
        ret1 = float(df["Close"].pct_change().iloc[-1])
        ma = float(df["Close"].rolling(config.BREAKER_MA_DAYS).mean().iloc[-1])

        result.close = close
        result.ret1 = ret1
        result.ma = ma

        result.cond_close_lt_ma = bool(np.isfinite(ma) and close < ma)
        result.cond_ret_lt_threshold = bool(np.isfinite(ret1) and ret1 < config.BREAKER_RET_THRESHOLD)

        if result.cond_close_lt_ma:
            logger.warning(f"⚠️ 地合い警戒: {config.BREAKER_TICKER} が {config.BREAKER_MA_DAYS}日線を下回っています")
        if result.cond_ret_lt_threshold:
            logger.warning(f"🚨 パニック警戒: {config.BREAKER_TICKER} が前日比 {ret1 * 100:.2f}% 急落")

        if not config.BREAKER_ENABLED:
            result.note = "breaker disabled in settings"
            return result

        result.breaker = result.cond_close_lt_ma or result.cond_ret_lt_threshold
        result.reason = build_breaker_reason(result.cond_close_lt_ma, result.cond_ret_lt_threshold)

        if result.breaker:
            result.note = "breaker triggered"
            logger.error("🛑 【ブレーカー発動】 本日の買いシグナルは強制的に停止(0件)になります。")

        return result
    except Exception as exc:
        logger.error(f"地合いチェックエラー: {exc}")
        result.note = str(exc)
        return result


def build_breaker_observation_rows(
    breaker_result: MarketBreakerResult,
    blocked_candidates: Optional[pd.DataFrame],
) -> tuple[list[Dict[str, object]], list[Dict[str, object]]]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    breaker_name = "market_breaker"
    event_rows = []
    simulated_rows = []

    if blocked_candidates is None or blocked_candidates.empty:
        event_rows.append(
            {
                "timestamp": timestamp,
                "symbol": "",
                "side_candidate": "",
                "intended_qty": "",
                "breaker_name": breaker_name,
                "breaker_reason": breaker_result.reason,
                "market_phase": "pre_market_batch",
                "price_reference": breaker_result.close if breaker_result.close is not None else "",
                "volatility_reference": breaker_result.ret1 if breaker_result.ret1 is not None else "",
                "spread_reference": "",
                "action_taken": "halt",
            }
        )
        return event_rows, simulated_rows

    for _, row in blocked_candidates.iterrows():
        symbol = str(row.get("TickerCode", ""))
        price_reference = float(row.get("Close", 0.0)) if pd.notna(row.get("Close")) else ""
        volatility_reference = (
            float(row.get("ATR_Prev_Ratio", 0.0)) if pd.notna(row.get("ATR_Prev_Ratio")) else ""
        )
        signal_score = float(row.get("Net_Score", 0.0) * 100.0) if pd.notna(row.get("Net_Score")) else ""

        event_rows.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "side_candidate": "BUY",
                "intended_qty": "",
                "breaker_name": breaker_name,
                "breaker_reason": breaker_result.reason,
                "market_phase": "pre_market_batch",
                "price_reference": price_reference,
                "volatility_reference": volatility_reference,
                "spread_reference": "",
                "action_taken": "skip_entry",
            }
        )

        simulated_rows.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": "BUY",
                "qty": "",
                "order_type": "MARKET",
                "intended_price": price_reference,
                "signal_score": signal_score,
                "model_decision": "recommendation_candidate",
                "blocked_by_breaker": True,
                "blocker_reason": breaker_result.reason,
                "would_open_or_close": "open",
            }
        )

    return event_rows, simulated_rows


def record_breaker_observations(
    breaker_result: MarketBreakerResult,
    blocked_candidates: Optional[pd.DataFrame],
) -> None:
    event_rows, simulated_rows = build_breaker_observation_rows(breaker_result, blocked_candidates)
    append_observation_rows(BREAKER_EVENT_LOG_PATH, BREAKER_EVENT_LOG_COLUMNS, event_rows)
    logger.info(f"BREAKER_EVENT_LOGGED count={len(event_rows)} path={BREAKER_EVENT_LOG_PATH}")

    if simulated_rows:
        append_observation_rows(SIMULATED_ORDER_LOG_PATH, SIMULATED_ORDER_LOG_COLUMNS, simulated_rows)
        logger.info(f"SIMULATED_ORDER_LOGGED count={len(simulated_rows)} path={SIMULATED_ORDER_LOG_PATH}")


def build_health_row(breaker_result: MarketBreakerResult) -> Dict[str, object]:
    return {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "breaker_enabled": breaker_result.breaker_enabled,
        "breaker": breaker_result.breaker,
        "reason": breaker_result.reason,
        "breaker_ticker": breaker_result.ticker,
        "ma_days": breaker_result.ma_days,
        "ret_threshold": breaker_result.ret_threshold,
        "cond_close_lt_ma": breaker_result.cond_close_lt_ma,
        "cond_ret_lt_threshold": breaker_result.cond_ret_lt_threshold,
        "topix_close": breaker_result.close if breaker_result.close is not None else "",
        "topix_ma": breaker_result.ma if breaker_result.ma is not None else "",
        "topix_ret1": breaker_result.ret1 if breaker_result.ret1 is not None else "",
        "note": breaker_result.note,
    }


def main() -> int:
    logger.info("=========================================")
    logger.info("🌅 毎朝のAI予測バッチ処理を開始します (V2.1 Net_Score)")
    logger.info("=========================================")

    config = BatchConfig.load()
    breaker_result = check_market_breaker(config)
    health_row = build_health_row(breaker_result)
    breaker_active = breaker_result.breaker

    try:
        try:
            ranker_model = joblib.load("models/ranker_model.pkl")
            classifier_model = joblib.load("models/classifier_model.pkl")
            meta_model = joblib.load("models/meta_model.pkl")
            reg_model = joblib.load("models/regressor_model.pkl")
            scaler = joblib.load("models/scaler.pkl")
            selected_features = joblib.load("models/selected_features.pkl")
            logger.info("✅ Kaggle産 学習済みAIモデルの読み込みに成功しました！")
        except Exception as exc:
            logger.error(f"❌ モデルの読み込みに失敗: {exc}")
            health_row["note"] = append_note(str(health_row.get("note", "")), f"model_load_failed: {exc}")
            return 1

        tickers = get_tickers()
        if not tickers:
            logger.error("❌ tickers.txt が空です")
            health_row["note"] = append_note(str(health_row.get("note", "")), "tickers_empty")
            return 1

        logger.info(f"📊 {len(tickers)} 銘柄の最新データを取得中...")
        macro = get_macro_data()

        stock_dict = {}
        for name, ticker in tickers.items():
            df = get_stock_features(ticker, macro)
            if df is not None and not df.empty:
                df["Ticker"] = name
                df["TickerCode"] = ticker.replace(".T", "")
                stock_dict[name] = df

        if not stock_dict:
            logger.error("❌ データ取得失敗")
            health_row["note"] = append_note(str(health_row.get("note", "")), "stock_data_empty")
            return 1

        rsi_df = pd.DataFrame({name: data["RSI_14"] for name, data in stock_dict.items()})
        rsi_mean, rsi_std = rsi_df.mean(axis=1), rsi_df.std(axis=1)

        df_all = []
        for _, df in stock_dict.items():
            stock_df = df.copy()
            stock_df["RSI_Z_Score"] = (stock_df["RSI_14"] - rsi_mean) / (rsi_std + 1e-9)
            df_all.append(stock_df)

        panel = pd.concat(df_all).sort_index()
        panel["ATR_Prev_Ratio"] = panel.groupby("Ticker")["ATR"].shift(1)

        all_features = [
            "Log_Return_Norm",
            "Frac_Diff_0.5",
            "Disparity_5_Norm",
            "Disparity_25_Norm",
            "SMA_Cross",
            "RSI_14",
            "BB_PctB",
            "BB_Bandwidth",
            "MACD_Norm",
            "ATR",
            "OBV_Ret",
            "USDJPY_Ret",
            "SP500_Ret",
            "TOPIX_Ret",
            "EMA_Diff_Ratio",
            "MACD_Div",
            "Log_Return_Norm_Lag1",
            "Log_Return_Norm_Lag2",
            "RSI_14_Lag1",
            "RSI_14_Lag2",
            "MACD_Norm_Lag1",
            "MACD_Norm_Lag2",
            "RSI_Z_Score",
            "Excess_Return",
            "Excess_Return_20d",
        ]

        panel = panel.dropna(subset=all_features + ["ATR_Prev_Ratio"])
        if panel.empty:
            logger.error("❌ 前処理後データが空")
            health_row["note"] = append_note(str(health_row.get("note", "")), "panel_empty")
            return 1

        latest_date = panel.index.max()
        latest = panel.loc[panel.index == latest_date].copy()
        logger.info(f"📅 予測対象日: {latest_date.strftime('%Y-%m-%d')} のデータで推論します")

        x_raw = latest[all_features]
        x_scaled = pd.DataFrame(scaler.transform(x_raw), index=x_raw.index, columns=all_features)
        x_sel = x_scaled[selected_features]

        logger.info("⚡ AIによる推論(Net Score計算)を実行中...")
        latest["Ranking_Score"] = ranker_model.predict(x_sel)
        base_prob = classifier_model.predict_proba(x_sel)[:, 1]
        latest["Prob_Up"] = base_prob

        x_meta = x_sel.copy()
        x_meta["Base_Prob"] = base_prob
        latest["Meta_Prob"] = meta_model.predict_proba(x_meta)[:, 1]
        latest["Pred_Return"] = reg_model.predict(x_sel)
        latest["cost_hat"] = latest["ATR_Prev_Ratio"].apply(cost_hat_roundtrip)
        latest["Net_Score"] = latest["Pred_Return"] - latest["cost_hat"]
        latest = latest[latest["Net_Score"] > 0].copy()

        if latest.empty:
            if breaker_active:
                record_breaker_observations(breaker_result, blocked_candidates=None)
                logger.warning("🛑 ブレーカー発動：recommendations.csv を空で出力します")
                write_empty_recommendations(config.RECO_CSV_PATH)
                health_row["note"] = append_note(str(health_row.get("note", "")), "breaker observation only")
                health_row["note"] = append_note(str(health_row.get("note", "")), "recommendations cleared by breaker")
                return 0
            logger.info("😴 本日は Net_Score>0 の銘柄がありませんでした")
            write_empty_recommendations(config.RECO_CSV_PATH)
            health_row["note"] = append_note(str(health_row.get("note", "")), "no_recommendations")
            return 0

        latest = latest.sort_values("Net_Score", ascending=False)

        if breaker_active:
            blocked_count = len(latest)
            record_breaker_observations(breaker_result, blocked_candidates=latest)
            logger.warning("🛑 ブレーカー発動：recommendations.csv を空で出力します")
            write_empty_recommendations(config.RECO_CSV_PATH)
            health_row["note"] = append_note(str(health_row.get("note", "")), f"breaker_blocked_candidates={blocked_count}")
            health_row["note"] = append_note(str(health_row.get("note", "")), "recommendations cleared by breaker")
            return 0

        results = []
        for index, (_, row) in enumerate(latest.iterrows()):
            reasons = []
            if index == 0:
                reasons.append("Net_Score第1位（コスト控除後でも期待値が残る）")
            reasons.append(
                f"Pred={row['Pred_Return'] * 100:.2f}%, "
                f"Cost={row['cost_hat'] * 100:.2f}% -> Net={row['Net_Score'] * 100:.2f}%"
            )

            results.append(
                {
                    "銘柄名": row["Ticker"],
                    "銘柄コード": row["TickerCode"],
                    "今日の終値": float(row["Close"]),
                    "Net_Score(%)": float(row["Net_Score"] * 100.0),
                    "予測リターン(%)": float(row["Pred_Return"] * 100.0),
                    "推定往復コスト(%)": float(row["cost_hat"] * 100.0),
                    "メタ確信度": float(row["Meta_Prob"] * 100.0),
                    "短期スコア": float(row["Net_Score"] * 100.0),
                    "おすすめ理由": " ".join(reasons),
                }
            )

        df_res = pd.DataFrame(results)
        df_res = df_res[[column for column in RECOMMENDATION_COLUMNS if column in df_res.columns]]

        ensure_parent_dir(config.RECO_CSV_PATH)
        df_res.to_csv(config.RECO_CSV_PATH, index=False, encoding="utf-8-sig")
        logger.info("✅ V2.1 recommendations.csv を更新しました（Net_Score採用）")
        health_row["note"] = append_note(str(health_row.get("note", "")), f"recommendations={len(df_res)}")

        history_file = "prediction_history.csv"
        df_hist = df_res.copy()
        df_hist.insert(0, "Date", latest_date.strftime("%Y-%m-%d"))

        if os.path.exists(history_file):
            df_hist.to_csv(history_file, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            df_hist.to_csv(history_file, index=False, encoding="utf-8-sig")
        logger.info("📓 prediction_history.csv に履歴を追記しました。")
        return 0
    except Exception as exc:
        logger.exception(f"❌ daily_batch.py 実行中に予期しないエラー: {exc}")
        health_row["note"] = append_note(str(health_row.get("note", "")), f"unexpected_error: {exc}")
        return 1
    finally:
        write_health_log(config.HEALTH_LOG_PATH, health_row)
        logger.info(f"🩺 daily_health_log.csv を更新しました: {config.HEALTH_LOG_PATH}")


if __name__ == "__main__":
    sys.exit(main())
