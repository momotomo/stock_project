import argparse
import csv
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None

HEALTH_LOG_PATH_DEFAULT = "daily_health_log.csv"
OUT_DIR_DEFAULT = "logs"
REQUIRED_COLUMNS = [
    "breaker_enabled",
    "ret_threshold",
    "cond_close_lt_ma",
    "cond_ret_lt_threshold",
]


@dataclass
class HealthLogRow:
    run_at: datetime
    breaker_enabled: bool
    breaker: bool
    reason: str
    breaker_ticker: str
    ma_days: int
    ret_threshold: float
    cond_close_lt_ma: bool
    cond_ret_lt_threshold: bool
    topix_close: Optional[float]
    topix_ma: Optional[float]
    topix_ret1: Optional[float]
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14, help="Lookback window in days")
    parser.add_argument("--out-dir", default=OUT_DIR_DEFAULT, help="Output directory for report files")
    parser.add_argument("--input", default=HEALTH_LOG_PATH_DEFAULT, help="Path to daily_health_log.csv")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None

    candidates = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_bool(value: str) -> Optional[bool]:
    normalized = (value or "").strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return None


def parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def read_health_log(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return [], 0, len(REQUIRED_COLUMNS), []

    valid_rows: List[HealthLogRow] = []
    skipped_rows = 0
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            return [], 0, 0, missing_columns

        for raw_row in reader:
            run_at = parse_datetime(raw_row.get("run_at", ""))
            breaker_enabled = parse_bool(raw_row.get("breaker_enabled", ""))
            breaker = parse_bool(raw_row.get("breaker", ""))
            ret_threshold = parse_float(raw_row.get("ret_threshold", ""))
            cond_close_lt_ma = parse_bool(raw_row.get("cond_close_lt_ma", ""))
            cond_ret_lt_threshold = parse_bool(raw_row.get("cond_ret_lt_threshold", ""))

            if None in {
                run_at,
                breaker_enabled,
                breaker,
                ret_threshold,
                cond_close_lt_ma,
                cond_ret_lt_threshold,
            }:
                skipped_rows += 1
                continue

            valid_rows.append(
                HealthLogRow(
                    run_at=run_at,
                    breaker_enabled=breaker_enabled,
                    breaker=breaker,
                    reason=(raw_row.get("reason") or "UNKNOWN").strip() or "UNKNOWN",
                    breaker_ticker=(raw_row.get("breaker_ticker") or "").strip(),
                    ma_days=parse_int(raw_row.get("ma_days")),
                    ret_threshold=ret_threshold,
                    cond_close_lt_ma=cond_close_lt_ma,
                    cond_ret_lt_threshold=cond_ret_lt_threshold,
                    topix_close=parse_float(raw_row.get("topix_close", "")),
                    topix_ma=parse_float(raw_row.get("topix_ma", "")),
                    topix_ret1=parse_float(raw_row.get("topix_ret1", "")),
                    note=(raw_row.get("note") or "").strip(),
                )
            )

    return valid_rows, skipped_rows, len(fieldnames), []


def filter_recent_rows(rows: List[HealthLogRow], days: int) -> List[HealthLogRow]:
    since_date = datetime.now().date() - timedelta(days=days)
    filtered = [row for row in rows if row.run_at.date() >= since_date]
    filtered.sort(key=lambda row: row.run_at, reverse=True)
    return filtered


def build_summary(rows: List[HealthLogRow], input_path: str, days: int, skipped_rows: int) -> Dict[str, object]:
    total_runs = len(rows)
    breaker_true_count = sum(1 for row in rows if row.breaker)
    cond_close_true_count = sum(1 for row in rows if row.cond_close_lt_ma)
    cond_ret_true_count = sum(1 for row in rows if row.cond_ret_lt_threshold)
    both_conditions_true_count = sum(
        1 for row in rows if row.cond_close_lt_ma and row.cond_ret_lt_threshold
    )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": input_path,
        "lookback_days": days,
        "total_runs": total_runs,
        "breaker_true_count": breaker_true_count,
        "breaker_true_rate": (breaker_true_count / total_runs) if total_runs else 0.0,
        "cond_close_lt_ma_true_count": cond_close_true_count,
        "cond_ret_lt_threshold_true_count": cond_ret_true_count,
        "both_conditions_true_count": both_conditions_true_count,
        "skipped_old_format_count": skipped_rows,
    }


def build_reason_counts(rows: List[HealthLogRow]) -> List[Dict[str, object]]:
    total_runs = len(rows)
    counter = Counter(row.reason for row in rows)
    reason_rows = []
    for reason, count in counter.most_common():
        reason_rows.append(
            {
                "reason": reason,
                "count": count,
                "rate": (count / total_runs) if total_runs else 0.0,
            }
        )
    return reason_rows


def build_daily_rows(rows: List[HealthLogRow]) -> List[Dict[str, object]]:
    daily_rows = []
    for row in rows:
        daily_rows.append(
            {
                "run_at": row.run_at.strftime("%Y-%m-%d %H:%M:%S"),
                "breaker": row.breaker,
                "reason": row.reason,
                "breaker_enabled": row.breaker_enabled,
                "breaker_ticker": row.breaker_ticker,
                "ma_days": row.ma_days,
                "ret_threshold": row.ret_threshold,
                "cond_close_lt_ma": row.cond_close_lt_ma,
                "cond_ret_lt_threshold": row.cond_ret_lt_threshold,
                "topix_close": "" if row.topix_close is None else row.topix_close,
                "topix_ma": "" if row.topix_ma is None else row.topix_ma,
                "topix_ret1": "" if row.topix_ret1 is None else row.topix_ret1,
                "note": row.note,
            }
        )
    return daily_rows


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def write_summary_text(path: str, summary: Dict[str, object], reason_counts: List[Dict[str, object]]) -> str:
    lines = [
        "Health Log Report",
        f"generated_at: {summary['generated_at']}",
        f"input_path: {summary['input_path']}",
        f"lookback_days: {summary['lookback_days']}",
        f"total_runs: {summary['total_runs']}",
        f"breaker_true_count: {summary['breaker_true_count']}",
        f"breaker_true_rate: {summary['breaker_true_rate']:.2%}",
        f"cond_close_lt_ma_true_count: {summary['cond_close_lt_ma_true_count']}",
        f"cond_ret_lt_threshold_true_count: {summary['cond_ret_lt_threshold_true_count']}",
        f"both_conditions_true_count: {summary['both_conditions_true_count']}",
        f"skipped_old_format_count: {summary['skipped_old_format_count']}",
        "reason_counts:",
    ]

    if reason_counts:
        for row in reason_counts:
            lines.append(f"  - {row['reason']}: {row['count']} ({row['rate']:.2%})")
    else:
        lines.append("  - none")

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    return text


def write_html_reports(out_dir: str, reason_counts: List[Dict[str, object]], daily_rows: List[Dict[str, object]]) -> None:
    if go is None:
        return

    if reason_counts:
        fig_reason = go.Figure(
            data=[
                go.Bar(
                    x=[row["reason"] for row in reason_counts],
                    y=[row["count"] for row in reason_counts],
                )
            ]
        )
        fig_reason.update_layout(title="Health Log Reason Counts", xaxis_title="reason", yaxis_title="count")
        fig_reason.write_html(os.path.join(out_dir, "health_reason_counts.html"))

    if daily_rows:
        fig_daily = go.Figure()
        fig_daily.add_trace(
            go.Scatter(
                x=[row["run_at"] for row in daily_rows],
                y=[1 if row["breaker"] else 0 for row in daily_rows],
                mode="lines+markers",
                text=[row["reason"] for row in daily_rows],
                name="breaker",
            )
        )
        fig_daily.update_layout(
            title="Health Log Daily Breaker Status",
            xaxis_title="run_at",
            yaxis_title="breaker",
            yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["False", "True"]),
        )
        fig_daily.write_html(os.path.join(out_dir, "health_daily.html"))


def main(days: int, out_dir: str, input_path: str) -> int:
    ensure_dir(out_dir)
    rows, skipped_rows, _, missing_columns = read_health_log(input_path)

    if missing_columns:
        summary = build_summary([], input_path, days, skipped_rows)
        summary["missing_required_columns"] = "|".join(missing_columns)
        reason_counts: List[Dict[str, object]] = []
        daily_rows: List[Dict[str, object]] = []
    else:
        filtered_rows = filter_recent_rows(rows, days)
        summary = build_summary(filtered_rows, input_path, days, skipped_rows)
        summary["missing_required_columns"] = ""
        reason_counts = build_reason_counts(filtered_rows)
        daily_rows = build_daily_rows(filtered_rows)

    summary_path = os.path.join(out_dir, "health_summary.csv")
    reason_counts_path = os.path.join(out_dir, "health_reason_counts.csv")
    daily_path = os.path.join(out_dir, "health_daily.csv")
    summary_txt_path = os.path.join(out_dir, "health_summary.txt")

    write_csv(summary_path, [summary], list(summary.keys()))
    write_csv(reason_counts_path, reason_counts, ["reason", "count", "rate"])
    write_csv(
        daily_path,
        daily_rows,
        [
            "run_at",
            "breaker",
            "reason",
            "breaker_enabled",
            "breaker_ticker",
            "ma_days",
            "ret_threshold",
            "cond_close_lt_ma",
            "cond_ret_lt_threshold",
            "topix_close",
            "topix_ma",
            "topix_ret1",
            "note",
        ],
    )
    summary_text = write_summary_text(summary_txt_path, summary, reason_counts)
    write_html_reports(out_dir, reason_counts, daily_rows)

    print(summary_text)
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args.days, args.out_dir, args.input))
