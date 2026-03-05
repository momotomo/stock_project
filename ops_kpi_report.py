import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px

# =========================================================
# 運用KPIレポート生成スクリプト (ops_kpi_report.py)
# 使い方: コマンドプロンプトで `python ops_kpi_report.py --days 14` を実行
# =========================================================

TRADE_LOG = "trade_execution_log.csv"
STATUS_LOG = "order_status_log.csv"
OUT_DIR = "logs"

def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")

def parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def main(days: int):
    os.makedirs(OUT_DIR, exist_ok=True)

    trade = read_csv_safe(TRADE_LOG)
    status = read_csv_safe(STATUS_LOG)

    today = datetime.now().date()
    since = today - timedelta(days=days)

    # ========= trade_execution_log (約定・発注ログ) =========
    if not trade.empty:
        trade["order_sent_time_dt"] = parse_dt(trade.get("order_sent_time"))
        trade["fill_time_dt"] = parse_dt(trade.get("fill_time"))
        trade["date"] = trade["order_sent_time_dt"].dt.date

        trade = trade[trade["date"] >= since].copy()

        # 遅延（ミリ秒）
        trade["latency_ms"] = (trade["fill_time_dt"] - trade["order_sent_time_dt"]).dt.total_seconds() * 1000.0

        # スリッページ率の計算 (Expectedに対するズレ)
        exp = np.where(trade["side"].astype(str).str.upper() == "SELL",
                       trade.get("expected_bid", np.nan),
                       trade.get("expected_ask", np.nan))
        exp = pd.to_numeric(exp, errors="coerce")
        slip_yen = pd.to_numeric(trade.get("slippage_yen", np.nan), errors="coerce")
        trade["slippage_pct"] = np.where(exp > 0, slip_yen / exp, np.nan)

        trade["spread_pct"] = pd.to_numeric(trade.get("spread_pct", np.nan), errors="coerce")

        # 日次サマリの集計
        daily_trade = trade.groupby("date").agg(
            trades=("order_id", "count"),
            symbols=("symbol", pd.Series.nunique),
            latency_ms_p50=("latency_ms", lambda x: np.nanpercentile(x.dropna(), 50) if x.notna().any() else np.nan),
            latency_ms_p95=("latency_ms", lambda x: np.nanpercentile(x.dropna(), 95) if x.notna().any() else np.nan),
            spread_bps_p50=("spread_pct", lambda x: (np.nanpercentile(x.dropna(), 50) * 10000) if x.notna().any() else np.nan),
            slip_bps_p50=("slippage_pct", lambda x: (np.nanpercentile(x.dropna(), 50) * 10000) if x.notna().any() else np.nan),
            slip_bps_p95=("slippage_pct", lambda x: (np.nanpercentile(x.dropna(), 95) * 10000) if x.notna().any() else np.nan),
        ).reset_index()

        daily_trade.to_csv(os.path.join(OUT_DIR, "kpi_daily_trade.csv"), index=False, encoding="utf-8-sig")

        # グラフの作成と保存
        fig_latency = px.histogram(trade, x="latency_ms", nbins=50, title="発注〜約定の遅延 (ms) 分布")
        fig_spread = px.histogram(trade, x="spread_pct", nbins=50, title="スプレッド率 (%) 分布")
        fig_slip = px.histogram(trade, x="slippage_pct", nbins=50, title="スリッページ率 (%) 分布")
        fig_daily = px.line(daily_trade, x="date", y=["latency_ms_p50", "latency_ms_p95", "spread_bps_p50", "slip_bps_p50", "slip_bps_p95"], title="日次KPI推移 (中央値/95パーセンタイル)")

        fig_latency.write_html(os.path.join(OUT_DIR, "kpi_latency_hist.html"))
        fig_spread.write_html(os.path.join(OUT_DIR, "kpi_spread_hist.html"))
        fig_slip.write_html(os.path.join(OUT_DIR, "kpi_slippage_hist.html"))
        fig_daily.write_html(os.path.join(OUT_DIR, "kpi_daily_timeseries.html"))

    # ========= order_status_log (エラー・タイムアウト等) =========
    if not status.empty:
        status["order_sent_time_dt"] = parse_dt(status.get("order_sent_time"))
        status["date"] = status["order_sent_time_dt"].dt.date
        status = status[status["date"] >= since].copy()

        status["status"] = status.get("status", "").astype(str)
        daily_status = status.groupby(["date", "status"]).size().reset_index(name="count")

        # TIMEOUT率の計算
        total_by_day = daily_status.groupby("date")["count"].sum().reset_index(name="total")
        timeout_by_day = daily_status[daily_status["status"].str.contains("TIMEOUT", na=False)].groupby("date")["count"].sum().reset_index(name="timeout")
        merged = total_by_day.merge(timeout_by_day, on="date", how="left").fillna({"timeout": 0})
        merged["timeout_rate"] = merged["timeout"] / merged["total"].replace(0, np.nan)

        merged.to_csv(os.path.join(OUT_DIR, "kpi_daily_status.csv"), index=False, encoding="utf-8-sig")

        fig_status = px.bar(daily_status, x="date", y="count", color="status", title="ステータス別 発生件数")
        fig_timeout = px.line(merged, x="date", y="timeout_rate", title="タイムアウト(未約定パージ)率 推移")

        fig_status.write_html(os.path.join(OUT_DIR, "kpi_status_counts.html"))
        fig_timeout.write_html(os.path.join(OUT_DIR, "kpi_timeout_rate.html"))

    print(f"✅ KPIレポートを出力しました。保存先: {OUT_DIR}/ (集計期間={days}日)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    main(args.days)