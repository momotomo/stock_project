# ops_kpi_report.py
# 使い方例:
#   本番ログだけ: python ops_kpi_report.py --env prod --days 14
#   SIMログだけ : python ops_kpi_report.py --env sim  --days 14
#
# 出力:
#   logs/kpi_*_{ENV}.html / logs/kpi_*_{ENV}.csv

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px

OUT_DIR_DEFAULT = "logs"

# ログファイル名（A案：完全分離）
TRADE_LOG_PROD = "trade_execution_log.csv"
STATUS_LOG_PROD = "order_status_log.csv"
TRADE_LOG_SIM = "trade_execution_log_SIM.csv"
STATUS_LOG_SIM = "order_status_log_SIM.csv"


def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def parse_dt(series: pd.Series) -> pd.Series:
    # "YYYY-mm-dd HH:MM:SS.mmm" を想定。空欄もあるのでcoerce
    return pd.to_datetime(series, errors="coerce")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_paths(env: str) -> tuple[str, str]:
    env = env.lower()
    if env == "prod":
        return TRADE_LOG_PROD, STATUS_LOG_PROD
    if env == "sim":
        return TRADE_LOG_SIM, STATUS_LOG_SIM
    raise ValueError("env must be 'prod' or 'sim'")


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def summarize_trade(trade: pd.DataFrame, since_date: datetime.date) -> tuple[pd.DataFrame, dict]:
    """
    trade_execution_log のKPI作成
    戻り値:
      - daily_trade: 日次サマリ（CSV用）
      - figs: plotly figures（html保存用）
    """
    figs = {}

    # 必須列がない場合は空返し
    if trade.empty or "order_sent_time" not in trade.columns:
        return pd.DataFrame(), figs

    # 日付・時刻
    trade = trade.copy()
    trade["order_sent_time_dt"] = parse_dt(trade["order_sent_time"])
    trade["fill_time_dt"] = parse_dt(trade["fill_time"]) if "fill_time" in trade.columns else pd.NaT
    trade["date"] = trade["order_sent_time_dt"].dt.date

    trade = trade[trade["date"] >= since_date].copy()
    if trade.empty:
        return pd.DataFrame(), figs

    # 遅延（ms）：fill_timeが無い/空の場合はNaN
    trade["latency_ms"] = (trade["fill_time_dt"] - trade["order_sent_time_dt"]).dt.total_seconds() * 1000.0

    # spread_pct / slippage_yen / expected_* の数値化
    if "spread_pct" in trade.columns:
        trade["spread_pct"] = safe_numeric(trade["spread_pct"])
    else:
        trade["spread_pct"] = np.nan

    if "slippage_yen" in trade.columns:
        trade["slippage_yen"] = safe_numeric(trade["slippage_yen"])
    else:
        trade["slippage_yen"] = np.nan

    # expectedの参照（BUYはAsk、SELLはBid）
    side = trade.get("side", "").astype(str).str.upper()
    expected_ask = safe_numeric(trade.get("expected_ask", np.nan))
    expected_bid = safe_numeric(trade.get("expected_bid", np.nan))
    expected = np.where(side == "SELL", expected_bid, expected_ask)
    expected = pd.to_numeric(expected, errors="coerce")

    # slippage_pct（ログ定義が「不利=+」になっている前提で、そのまま比率に）
    trade["slippage_pct"] = np.where(expected > 0, trade["slippage_yen"] / expected, np.nan)

    # 日次サマリ
    def pctile(x: pd.Series, q: int) -> float:
        x = x.dropna()
        if x.empty:
            return float("nan")
        return float(np.nanpercentile(x, q))

    daily_trade = trade.groupby("date").agg(
        trades=("order_id", "count") if "order_id" in trade.columns else ("latency_ms", "count"),
        symbols=("symbol", pd.Series.nunique) if "symbol" in trade.columns else ("latency_ms", "count"),
        latency_ms_p50=("latency_ms", lambda x: pctile(x, 50)),
        latency_ms_p95=("latency_ms", lambda x: pctile(x, 95)),
        spread_bps_p50=("spread_pct", lambda x: pctile(x, 50) * 10000),
        spread_bps_p95=("spread_pct", lambda x: pctile(x, 95) * 10000),
        slip_bps_p50=("slippage_pct", lambda x: pctile(x, 50) * 10000),
        slip_bps_p95=("slippage_pct", lambda x: pctile(x, 95) * 10000),
    ).reset_index()

    # 図：分布
    figs["latency_hist"] = px.histogram(trade, x="latency_ms", nbins=50, title="Latency (ms) distribution")
    figs["spread_hist"] = px.histogram(trade, x="spread_pct", nbins=50, title="Spread (%) distribution")
    figs["slip_hist"] = px.histogram(trade, x="slippage_pct", nbins=50, title="Slippage (%) distribution (as logged)")

    # 図：日次推移
    figs["daily_ts"] = px.line(
        daily_trade,
        x="date",
        y=["latency_ms_p50", "latency_ms_p95", "spread_bps_p50", "slip_bps_p50", "slip_bps_p95"],
        title="Daily KPIs"
    )

    return daily_trade, figs


def summarize_status(status: pd.DataFrame, since_date: datetime.date) -> tuple[pd.DataFrame, dict]:
    """
    order_status_log のKPI作成（TTL率など）
    戻り値:
      - daily_status_summary: 日次集計（CSV用）
      - figs: plotly figures（html保存用）
    """
    figs = {}
    if status.empty or "order_sent_time" not in status.columns:
        return pd.DataFrame(), figs

    status = status.copy()
    status["order_sent_time_dt"] = parse_dt(status["order_sent_time"])
    status["date"] = status["order_sent_time_dt"].dt.date

    status = status[status["date"] >= since_date].copy()
    if status.empty:
        return pd.DataFrame(), figs

    status["status"] = status.get("status", "").astype(str)

    daily_counts = status.groupby(["date", "status"]).size().reset_index(name="count")

    # TIMEOUT率（その日発生したステータスログに対する比率）
    total_by_day = daily_counts.groupby("date")["count"].sum().reset_index(name="total")
    timeout_by_day = daily_counts[daily_counts["status"].str.contains("TIMEOUT", na=False)].groupby("date")["count"].sum().reset_index(name="timeout")
    merged = total_by_day.merge(timeout_by_day, on="date", how="left").fillna({"timeout": 0})
    merged["timeout_rate"] = merged["timeout"] / merged["total"].replace(0, np.nan)

    # 図
    figs["status_counts"] = px.bar(daily_counts, x="date", y="count", color="status", title="Order Status counts by day")
    figs["timeout_rate"] = px.line(merged, x="date", y="timeout_rate", title="Timeout rate by day")

    # CSVは merged を返す（timeout_rateが使いやすい）
    return merged, figs


def write_figs(figs: dict, out_dir: str, prefix: str) -> None:
    for name, fig in figs.items():
        fig.write_html(os.path.join(out_dir, f"{prefix}_{name}.html"))


def main(env: str, days: int, out_dir: str):
    trade_path, status_path = get_paths(env)

    trade = read_csv_safe(trade_path)
    status = read_csv_safe(status_path)

    today = datetime.now().date()
    since = today - timedelta(days=days)

    ensure_dir(out_dir)
    env_tag = env.upper()

    # trade KPIs
    daily_trade, trade_figs = summarize_trade(trade, since)
    if not daily_trade.empty:
        daily_trade.to_csv(os.path.join(out_dir, f"kpi_daily_trade_{env_tag}.csv"), index=False, encoding="utf-8-sig")
        write_figs(trade_figs, out_dir, f"kpi_{env_tag}")

    # status KPIs
    daily_status, status_figs = summarize_status(status, since)
    if not daily_status.empty:
        daily_status.to_csv(os.path.join(out_dir, f"kpi_daily_status_{env_tag}.csv"), index=False, encoding="utf-8-sig")
        write_figs(status_figs, out_dir, f"kpi_{env_tag}")

    print(f"✅ KPI report generated: env={env_tag}, lookback={days} days, out_dir={out_dir}")
    print(f"   trade_log = {trade_path} ({'OK' if not trade.empty else 'EMPTY'})")
    print(f"   status_log= {status_path} ({'OK' if not status.empty else 'EMPTY'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["prod", "sim"], default="prod", help="Which logs to analyze (prod or sim)")
    ap.add_argument("--days", type=int, default=14, help="Lookback window in days")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT, help="Output directory for csv/html")
    args = ap.parse_args()
    main(args.env, args.days, args.out_dir)