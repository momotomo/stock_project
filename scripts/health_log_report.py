import argparse
import csv
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    go = None
    make_subplots = None

HEALTH_LOG_PATH_DEFAULT = "daily_health_log.csv"
OUT_DIR_DEFAULT = "logs"
NEW_FORMAT_COLUMNS = [
    "breaker_enabled",
    "ret_threshold",
    "cond_close_lt_ma",
    "cond_ret_lt_threshold",
]
DAILY_FIELDNAMES = [
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


@dataclass
class ReadResult:
    rows: List[HealthLogRow]
    input_status: str
    total_rows_read: int
    skipped_rows: int
    missing_header_columns: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate daily_health_log.csv for breaker analysis")
    parser.add_argument("--days", type=int, default=14, help="Lookback window in days")
    parser.add_argument("--input", default=HEALTH_LOG_PATH_DEFAULT, help="Path to daily_health_log.csv")
    parser.add_argument("--out-dir", default=OUT_DIR_DEFAULT, help="Output directory for report files")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_days(days: int) -> int:
    return max(days, 1)


def parse_datetime(value: str) -> Optional[datetime]:
    normalized = (value or "").strip()
    if not normalized:
        return None

    candidates = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def parse_bool(value: object) -> Optional[bool]:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes", "on", "y", "t"}:
        return True
    if normalized in {"false", "0", "no", "off", "n", "f"}:
        return False
    return None


def parse_float(value: object) -> Optional[float]:
    normalized = str(value or "").strip().replace(",", "")
    if not normalized:
        return None
    try:
        return float(normalized)
    except (TypeError, ValueError):
        return None


def parse_int(value: object) -> int:
    parsed = parse_float(value)
    if parsed is None:
        return 0
    return int(parsed)


def parse_optional_text(value: object, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def read_health_log(path: str) -> ReadResult:
    if not os.path.exists(path):
        return ReadResult([], "missing", 0, 0, [])
    if os.path.getsize(path) == 0:
        return ReadResult([], "empty", 0, 0, [])

    valid_rows: List[HealthLogRow] = []
    skipped_rows = 0
    total_rows_read = 0

    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_header_columns = [column for column in NEW_FORMAT_COLUMNS if column not in fieldnames]

        for raw_row in reader:
            total_rows_read += 1

            run_at = parse_datetime(raw_row.get("run_at"))
            breaker_enabled = parse_bool(raw_row.get("breaker_enabled"))
            breaker = parse_bool(raw_row.get("breaker"))
            ret_threshold = parse_float(raw_row.get("ret_threshold"))
            cond_close_lt_ma = parse_bool(raw_row.get("cond_close_lt_ma"))
            cond_ret_lt_threshold = parse_bool(raw_row.get("cond_ret_lt_threshold"))

            if None in (
                run_at,
                breaker_enabled,
                breaker,
                ret_threshold,
                cond_close_lt_ma,
                cond_ret_lt_threshold,
            ):
                skipped_rows += 1
                continue

            valid_rows.append(
                HealthLogRow(
                    run_at=run_at,
                    breaker_enabled=breaker_enabled,
                    breaker=breaker,
                    reason=parse_optional_text(raw_row.get("reason"), "UNKNOWN"),
                    breaker_ticker=parse_optional_text(raw_row.get("breaker_ticker")),
                    ma_days=parse_int(raw_row.get("ma_days")),
                    ret_threshold=ret_threshold,
                    cond_close_lt_ma=cond_close_lt_ma,
                    cond_ret_lt_threshold=cond_ret_lt_threshold,
                    topix_close=parse_float(raw_row.get("topix_close")),
                    topix_ma=parse_float(raw_row.get("topix_ma")),
                    topix_ret1=parse_float(raw_row.get("topix_ret1")),
                    note=parse_optional_text(raw_row.get("note")),
                )
            )

    return ReadResult(valid_rows, "ok", total_rows_read, skipped_rows, missing_header_columns)


def filter_recent_rows(rows: Sequence[HealthLogRow], days: int, now: datetime) -> Tuple[List[HealthLogRow], datetime]:
    lookback_days = normalize_days(days)
    since = now - timedelta(days=lookback_days)
    filtered = [row for row in rows if row.run_at >= since]
    filtered.sort(key=lambda row: row.run_at, reverse=True)
    return filtered, since


def build_summary(
    rows: Sequence[HealthLogRow],
    read_result: ReadResult,
    input_path: str,
    days: int,
    now: datetime,
    period_start: datetime,
) -> Dict[str, object]:
    total_runs = len(rows)
    breaker_true_count = sum(1 for row in rows if row.breaker)
    cond_close_true_count = sum(1 for row in rows if row.cond_close_lt_ma)
    cond_ret_true_count = sum(1 for row in rows if row.cond_ret_lt_threshold)
    both_conditions_true_count = sum(
        1 for row in rows if row.cond_close_lt_ma and row.cond_ret_lt_threshold
    )

    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": input_path,
        "target_period_start": period_start.strftime("%Y-%m-%d %H:%M:%S"),
        "target_period_end": now.strftime("%Y-%m-%d %H:%M:%S"),
        "lookback_days": normalize_days(days),
        "input_status": read_result.input_status,
        "total_rows_read": read_result.total_rows_read,
        "skipped_rows": read_result.skipped_rows,
        "missing_header_columns": "|".join(read_result.missing_header_columns),
        "total_runs": total_runs,
        "breaker_true_count": breaker_true_count,
        "breaker_true_rate": (breaker_true_count / total_runs) if total_runs else 0.0,
        "cond_close_lt_ma_true_count": cond_close_true_count,
        "cond_ret_lt_threshold_true_count": cond_ret_true_count,
        "both_conditions_true_count": both_conditions_true_count,
    }


def build_reason_counts(rows: Sequence[HealthLogRow]) -> List[Dict[str, object]]:
    counter = Counter(row.reason for row in rows)
    return [{"reason": reason, "count": count} for reason, count in counter.most_common()]


def build_daily_rows(rows: Sequence[HealthLogRow]) -> List[Dict[str, object]]:
    return [
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
        for row in rows
    ]


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def build_summary_rows(summary: Dict[str, object]) -> List[Dict[str, object]]:
    ordered_metrics = [
        "generated_at",
        "input_path",
        "target_period_start",
        "target_period_end",
        "lookback_days",
        "input_status",
        "total_rows_read",
        "skipped_rows",
        "missing_header_columns",
        "total_runs",
        "breaker_true_count",
        "breaker_true_rate",
        "cond_close_lt_ma_true_count",
        "cond_ret_lt_threshold_true_count",
        "both_conditions_true_count",
    ]
    rows = []
    for metric in ordered_metrics:
        value = summary.get(metric, "")
        rows.append({"metric": metric, "value": value})
    return rows


def write_summary_text(
    path: str,
    summary: Dict[str, object],
    reason_counts: Sequence[Dict[str, object]],
) -> str:
    total_runs = int(summary["total_runs"])
    lines = [
        "Health Log Report",
        f"generated_at: {summary['generated_at']}",
        f"input_path: {summary['input_path']}",
        f"target_period: {summary['target_period_start']} -> {summary['target_period_end']}",
        f"lookback_days: {summary['lookback_days']}",
        f"input_status: {summary['input_status']}",
        f"total_rows_read: {summary['total_rows_read']}",
        f"skipped_rows: {summary['skipped_rows']}",
    ]

    if summary["missing_header_columns"]:
        lines.append(f"missing_header_columns: {summary['missing_header_columns']}")

    lines.extend(
        [
            f"total_runs: {total_runs}",
            f"breaker_true_count: {summary['breaker_true_count']}",
            f"breaker_true_rate: {float(summary['breaker_true_rate']):.2%}",
            f"cond_close_lt_ma_true_count: {summary['cond_close_lt_ma_true_count']}",
            f"cond_ret_lt_threshold_true_count: {summary['cond_ret_lt_threshold_true_count']}",
            f"both_conditions_true_count: {summary['both_conditions_true_count']}",
            "reason_counts:",
        ]
    )

    if reason_counts:
        for row in reason_counts[:10]:
            rate = (row["count"] / total_runs) if total_runs else 0.0
            lines.append(f"  - {row['reason']}: {row['count']} ({rate:.2%})")
    else:
        lines.append("  - none")

    if summary["input_status"] == "missing":
        lines.append("note: input file was not found; empty report files were created.")
    elif summary["input_status"] == "empty":
        lines.append("note: input file was empty; empty report files were created.")
    elif summary["missing_header_columns"]:
        lines.append("note: required new-format columns were missing from the header; rows were skipped.")

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    return text


def write_html_reports(
    out_dir: str,
    reason_counts: Sequence[Dict[str, object]],
    rows: Sequence[HealthLogRow],
) -> None:
    if go is None or make_subplots is None:
        return

    if reason_counts:
        fig_reason = go.Figure(
            data=[
                go.Bar(
                    x=[row["reason"] for row in reason_counts],
                    y=[row["count"] for row in reason_counts],
                    marker_color="#1f77b4",
                )
            ]
        )
        fig_reason.update_layout(title="Health Log Reason Counts", xaxis_title="reason", yaxis_title="count")
        fig_reason.write_html(os.path.join(out_dir, "health_reason_counts.html"))

    if not rows:
        return

    x_values = [row.run_at for row in rows]
    breaker_values = [1 if row.breaker else 0 for row in rows]
    breaker_colors = ["#d62728" if row.breaker else "#2ca02c" for row in rows]
    hover_text = [
        f"reason={row.reason}<br>close_lt_ma={row.cond_close_lt_ma}<br>ret_lt_threshold={row.cond_ret_lt_threshold}"
        for row in rows
    ]

    fig_daily = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Breaker Status", "TOPIX Close vs MA", "TOPIX Return 1D"),
    )
    fig_daily.add_trace(
        go.Bar(
            x=x_values,
            y=breaker_values,
            marker_color=breaker_colors,
            text=[row.reason for row in rows],
            hovertext=hover_text,
            name="breaker",
        ),
        row=1,
        col=1,
    )
    fig_daily.add_trace(
        go.Scatter(x=x_values, y=[row.topix_close for row in rows], mode="lines+markers", name="topix_close"),
        row=2,
        col=1,
    )
    fig_daily.add_trace(
        go.Scatter(x=x_values, y=[row.topix_ma for row in rows], mode="lines+markers", name="topix_ma"),
        row=2,
        col=1,
    )
    fig_daily.add_trace(
        go.Bar(x=x_values, y=[row.topix_ret1 for row in rows], name="topix_ret1", marker_color="#ff7f0e"),
        row=3,
        col=1,
    )
    fig_daily.update_yaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["False", "True"],
        row=1,
        col=1,
    )
    fig_daily.update_layout(title="Health Log Daily Overview", showlegend=True)
    fig_daily.write_html(os.path.join(out_dir, "health_daily.html"))


def main(days: int, out_dir: str, input_path: str) -> int:
    ensure_dir(out_dir)
    now = datetime.now()
    read_result = read_health_log(input_path)
    filtered_rows, period_start = filter_recent_rows(read_result.rows, days, now)
    summary = build_summary(filtered_rows, read_result, input_path, days, now, period_start)
    reason_counts = build_reason_counts(filtered_rows)
    daily_rows = build_daily_rows(filtered_rows)

    summary_path = os.path.join(out_dir, "health_summary.csv")
    reason_counts_path = os.path.join(out_dir, "health_reason_counts.csv")
    daily_path = os.path.join(out_dir, "health_daily.csv")
    summary_txt_path = os.path.join(out_dir, "health_summary.txt")

    write_csv(summary_path, build_summary_rows(summary), ["metric", "value"])
    write_csv(reason_counts_path, reason_counts, ["reason", "count"])
    write_csv(daily_path, daily_rows, DAILY_FIELDNAMES)
    summary_text = write_summary_text(summary_txt_path, summary, reason_counts)
    write_html_reports(out_dir, reason_counts, filtered_rows)

    print(summary_text)
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args.days, args.out_dir, args.input))
