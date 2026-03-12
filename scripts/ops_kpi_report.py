import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_paths import (
    ORDER_STATUS_LOG_PATH,
    ORDER_STATUS_LOG_SIM_PATH,
    REPORTS_OUTPUT_DIR,
    TRADE_EXECUTION_LOG_PATH,
    TRADE_EXECUTION_LOG_SIM_PATH,
)

OUT_DIR_DEFAULT = REPORTS_OUTPUT_DIR
TRADE_LOG_PROD = TRADE_EXECUTION_LOG_PATH
STATUS_LOG_PROD = ORDER_STATUS_LOG_PATH
TRADE_LOG_SIM = TRADE_EXECUTION_LOG_SIM_PATH
STATUS_LOG_SIM = ORDER_STATUS_LOG_SIM_PATH


def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_paths(env: str):
    env = env.lower()
    if env == "prod":
        return TRADE_LOG_PROD, STATUS_LOG_PROD
    if env == "sim":
        return TRADE_LOG_SIM, STATUS_LOG_SIM
    raise ValueError("env must be 'prod' or 'sim'")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def get_series(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def percentile(series: pd.Series, q: int) -> float:
    series = series.dropna()
    if series.empty:
        return float("nan")
    return float(np.nanpercentile(series, q))


def summarize_trade(trade: pd.DataFrame, since_date: datetime.date):
    figures = {}
    if trade.empty or "order_sent_time" not in trade.columns:
        return pd.DataFrame(), figures

    trade = trade.copy()
    trade["order_sent_time_dt"] = parse_dt(trade["order_sent_time"])
    trade["fill_time_dt"] = parse_dt(trade["fill_time"]) if "fill_time" in trade.columns else pd.NaT
    trade["date"] = trade["order_sent_time_dt"].dt.date
    trade = trade[trade["date"] >= since_date].copy()
    if trade.empty:
        return pd.DataFrame(), figures

    trade["latency_ms"] = (trade["fill_time_dt"] - trade["order_sent_time_dt"]).dt.total_seconds() * 1000.0
    trade["spread_pct"] = safe_numeric(get_series(trade, "spread_pct"))
    trade["slippage_yen"] = safe_numeric(get_series(trade, "slippage_yen"))

    side = get_series(trade, "side", "").astype(str).str.upper()
    expected_ask = safe_numeric(get_series(trade, "expected_ask"))
    expected_bid = safe_numeric(get_series(trade, "expected_bid"))
    expected = np.where(side == "SELL", expected_bid, expected_ask)
    expected = pd.to_numeric(expected, errors="coerce")
    trade["slippage_pct"] = np.where(expected > 0, trade["slippage_yen"] / expected, np.nan)

    daily_trade = trade.groupby("date").agg(
        trades=("order_id", "count") if "order_id" in trade.columns else ("latency_ms", "count"),
        symbols=("symbol", pd.Series.nunique) if "symbol" in trade.columns else ("latency_ms", "count"),
        latency_ms_p50=("latency_ms", lambda s: percentile(s, 50)),
        latency_ms_p95=("latency_ms", lambda s: percentile(s, 95)),
        spread_bps_p50=("spread_pct", lambda s: percentile(s, 50) * 10000),
        spread_bps_p95=("spread_pct", lambda s: percentile(s, 95) * 10000),
        slip_bps_p50=("slippage_pct", lambda s: percentile(s, 50) * 10000),
        slip_bps_p95=("slippage_pct", lambda s: percentile(s, 95) * 10000),
    ).reset_index()

    figures["latency_hist"] = px.histogram(trade, x="latency_ms", nbins=50, title="Latency (ms) distribution")
    figures["spread_hist"] = px.histogram(trade, x="spread_pct", nbins=50, title="Spread (%) distribution")
    figures["slip_hist"] = px.histogram(trade, x="slippage_pct", nbins=50, title="Slippage (%) distribution")
    figures["daily_ts"] = px.line(
        daily_trade,
        x="date",
        y=["latency_ms_p50", "latency_ms_p95", "spread_bps_p50", "slip_bps_p50", "slip_bps_p95"],
        title="Daily KPIs",
    )
    return daily_trade, figures


def summarize_status(status: pd.DataFrame, since_date: datetime.date):
    figures = {}
    if status.empty or "order_sent_time" not in status.columns:
        return pd.DataFrame(), figures

    status = status.copy()
    status["order_sent_time_dt"] = parse_dt(status["order_sent_time"])
    status["date"] = status["order_sent_time_dt"].dt.date
    status = status[status["date"] >= since_date].copy()
    if status.empty:
        return pd.DataFrame(), figures

    status["status"] = get_series(status, "status", "").astype(str)
    daily_counts = status.groupby(["date", "status"]).size().reset_index(name="count")
    total_by_day = daily_counts.groupby("date")["count"].sum().reset_index(name="total")
    timeout_by_day = (
        daily_counts[daily_counts["status"].str.contains("TIMEOUT", na=False)]
        .groupby("date")["count"]
        .sum()
        .reset_index(name="timeout")
    )
    merged = total_by_day.merge(timeout_by_day, on="date", how="left").fillna({"timeout": 0})
    merged["timeout_rate"] = merged["timeout"] / merged["total"].replace(0, np.nan)

    figures["status_counts"] = px.bar(daily_counts, x="date", y="count", color="status", title="Order Status counts by day")
    figures["timeout_rate"] = px.line(merged, x="date", y="timeout_rate", title="Timeout rate by day")
    return merged, figures


def write_figures(figures: dict, out_dir: str, prefix: str) -> None:
    for name, figure in figures.items():
        figure.write_html(os.path.join(out_dir, f"{prefix}_{name}.html"))


def write_summary(out_dir: str, env_tag: str, days: int, trade: pd.DataFrame, status: pd.DataFrame, daily_trade: pd.DataFrame, daily_status: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        [
            {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "env": env_tag,
                "lookback_days": days,
                "trade_log_rows": len(trade),
                "status_log_rows": len(status),
                "trade_days": len(daily_trade),
                "status_days": len(daily_status),
            }
        ]
    )
    summary_csv = os.path.join(out_dir, f"kpi_summary_{env_tag}.csv")
    summary_html = os.path.join(out_dir, f"kpi_summary_{env_tag}.html")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    with open(summary_html, "w", encoding="utf-8") as handle:
        handle.write(summary.to_html(index=False))


def main(env: str, days: int, out_dir: str) -> None:
    trade_path, status_path = get_paths(env)
    trade = read_csv_safe(trade_path)
    status = read_csv_safe(status_path)
    since = datetime.now().date() - timedelta(days=days)

    ensure_dir(out_dir)
    env_tag = env.upper()

    daily_trade, trade_figures = summarize_trade(trade, since)
    daily_status, status_figures = summarize_status(status, since)

    daily_trade.to_csv(os.path.join(out_dir, f"kpi_daily_trade_{env_tag}.csv"), index=False, encoding="utf-8-sig")
    daily_status.to_csv(os.path.join(out_dir, f"kpi_daily_status_{env_tag}.csv"), index=False, encoding="utf-8-sig")
    write_summary(out_dir, env_tag, days, trade, status, daily_trade, daily_status)

    if trade_figures:
        write_figures(trade_figures, out_dir, f"kpi_{env_tag}")
    if status_figures:
        write_figures(status_figures, out_dir, f"kpi_{env_tag}")

    print(f"✅ KPI report generated: env={env_tag}, lookback={days} days, out_dir={out_dir}")
    print(f"   trade_log = {trade_path} ({'OK' if not trade.empty else 'EMPTY'})")
    print(f"   status_log= {status_path} ({'OK' if not status.empty else 'EMPTY'})")


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["prod", "sim"], default="prod", help="Which logs to analyze (prod or sim)")
    parser.add_argument("--days", type=int, default=14, help="Lookback window in days")
    parser.add_argument("--out-dir", default=OUT_DIR_DEFAULT, help="Output directory for csv/html")
    args = parser.parse_args()
    main(args.env, args.days, args.out_dir)


if __name__ == "__main__":
    cli()
