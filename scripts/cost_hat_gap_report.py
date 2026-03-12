import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_paths import (
    REPORTS_OUTPUT_DIR,
    TRADE_EXECUTION_LOG_PATH,
    TRADE_EXECUTION_LOG_SIM_PATH,
)

OUT_DIR_DEFAULT = REPORTS_OUTPUT_DIR
TRADE_LOG_PROD = TRADE_EXECUTION_LOG_PATH
TRADE_LOG_SIM = TRADE_EXECUTION_LOG_SIM_PATH
SUMMARY_PATH = "cost_hat_gap_summary.csv"
SUMMARY_TEXT_PATH = "cost_hat_gap_summary.txt"
BY_TIME_BUCKET_PATH = "cost_hat_gap_by_time_bucket.csv"
BY_SYMBOL_PATH = "cost_hat_gap_by_symbol.csv"
BY_ENTRY_EXIT_PATH = "cost_hat_gap_by_entry_exit.csv"
BY_FORCE_EXIT_PATH = "cost_hat_gap_by_force_exit.csv"
SUMMARY_COLUMNS = ["metric", "value"]
GROUP_METRIC_COLUMNS = [
    "count",
    "avg_actual_cost_bps",
    "avg_current_cost_hat_bps",
    "avg_cost_gap_bps",
    "median_cost_gap_bps",
    "avg_slippage_bps",
    "avg_spread_pct",
    "avg_turnover_proxy",
]
DEFAULT_BASE_FEE_BPS = 10.0
DEFAULT_SLIPPAGE_FACTOR = 0.05
DEFAULT_CLOSE_TIMING_PENALTY_BPS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare current cost_hat assumptions against realized execution costs")
    parser.add_argument("--env", choices=["prod", "sim"], default="prod", help="Which execution log to analyze")
    parser.add_argument("--days", type=int, default=14, help="Lookback window in days")
    parser.add_argument("--input", default="", help="Optional explicit input csv path")
    parser.add_argument("--out-dir", default=OUT_DIR_DEFAULT, help="Output directory for report files")
    parser.add_argument("--base-fee-bps", type=float, default=DEFAULT_BASE_FEE_BPS, help="Assumed one-way fee in bps")
    parser.add_argument(
        "--slippage-factor",
        type=float,
        default=DEFAULT_SLIPPAGE_FACTOR,
        help="Assumed coefficient for spread-based slippage proxy",
    )
    parser.add_argument(
        "--close-timing-penalty-bps",
        type=float,
        default=DEFAULT_CLOSE_TIMING_PENALTY_BPS,
        help="Penalty added for exit/preclose/force-exit rows",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_days(days: int) -> int:
    return max(int(days or 0), 1)


def get_default_input_path(env: str) -> str:
    return TRADE_LOG_SIM if env.lower() == "sim" else TRADE_LOG_PROD


def parse_datetime(value: object) -> Optional[datetime]:
    normalized = str(value or "").strip()
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


def parse_float(value: object) -> Optional[float]:
    normalized = str(value or "").strip().replace(",", "")
    if not normalized:
        return None
    try:
        return float(normalized)
    except (TypeError, ValueError):
        return None


def parse_bool(value: object) -> Optional[bool]:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes", "on", "y", "t"}:
        return True
    if normalized in {"false", "0", "no", "off", "n", "f"}:
        return False
    return None


def parse_text(value: object, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def read_csv_safe(path: str) -> Tuple[List[Dict[str, str]], str]:
    if not os.path.exists(path):
        return [], "missing"
    if os.path.getsize(path) == 0:
        return [], "empty"

    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), "ok"


def choose_event_time(row: Dict[str, str]) -> Optional[datetime]:
    return parse_datetime(row.get("fill_time")) or parse_datetime(row.get("order_sent_time"))


def should_apply_close_timing_penalty(entry_or_exit: str, time_bucket: str, is_force_exit: bool) -> bool:
    return entry_or_exit == "exit" or time_bucket == "preclose" or is_force_exit


def compute_cost_metrics(
    slippage_bps: float,
    spread_pct: float,
    entry_or_exit: str,
    time_bucket: str,
    is_force_exit: bool,
    base_fee_bps: float,
    slippage_factor: float,
    close_timing_penalty_bps: float,
) -> Tuple[float, float, float]:
    spread_bps = spread_pct * 10000.0

    # actual_cost_bps is the one-way realized cost proxy from execution logs:
    # fee + observed spread + adverse-positive realized slippage.
    actual_cost_bps = base_fee_bps + spread_bps + slippage_bps

    # current_cost_hat_bps is a simplified proxy of the current ATR-based cost_hat.
    # ATR is not logged yet, so spread_pct is used as a volatility/slippage proxy here.
    current_cost_hat_bps = base_fee_bps + (spread_bps * slippage_factor)
    if should_apply_close_timing_penalty(entry_or_exit, time_bucket, is_force_exit):
        current_cost_hat_bps += close_timing_penalty_bps

    cost_gap_bps = current_cost_hat_bps - actual_cost_bps
    return actual_cost_bps, current_cost_hat_bps, cost_gap_bps


def prepare_rows(
    raw_rows: Iterable[Dict[str, str]],
    days: int,
    now: datetime,
    base_fee_bps: float,
    slippage_factor: float,
    close_timing_penalty_bps: float,
) -> Tuple[List[Dict[str, object]], datetime]:
    since = now - timedelta(days=normalize_days(days))
    prepared: List[Dict[str, object]] = []

    for raw_row in raw_rows:
        event_time = choose_event_time(raw_row)
        if event_time is None or event_time < since:
            continue

        slippage_bps = parse_float(raw_row.get("slippage_bps"))
        slippage_pct = parse_float(raw_row.get("slippage_pct"))
        slippage_yen = parse_float(raw_row.get("slippage_yen"))
        expected_side_price = parse_float(raw_row.get("expected_side_price"))
        spread_pct = parse_float(raw_row.get("spread_pct"))

        if slippage_bps is None and slippage_pct is not None:
            slippage_bps = slippage_pct * 10000.0
        if slippage_bps is None and slippage_yen is not None and expected_side_price and expected_side_price > 0:
            slippage_bps = (slippage_yen / expected_side_price) * 10000.0
        if slippage_bps is None:
            continue

        entry_or_exit = parse_text(raw_row.get("entry_or_exit"), "unknown")
        time_bucket = parse_text(raw_row.get("time_bucket"), "unknown")
        is_force_exit = parse_bool(raw_row.get("is_force_exit")) or False
        spread_pct = spread_pct or 0.0

        actual_cost_bps, current_cost_hat_bps, cost_gap_bps = compute_cost_metrics(
            slippage_bps=slippage_bps,
            spread_pct=spread_pct,
            entry_or_exit=entry_or_exit,
            time_bucket=time_bucket,
            is_force_exit=is_force_exit,
            base_fee_bps=base_fee_bps,
            slippage_factor=slippage_factor,
            close_timing_penalty_bps=close_timing_penalty_bps,
        )

        prepared.append(
            {
                "event_time": event_time,
                "symbol": parse_text(raw_row.get("symbol"), "UNKNOWN"),
                "entry_or_exit": entry_or_exit,
                "time_bucket": time_bucket,
                "is_force_exit": is_force_exit,
                "slippage_bps": slippage_bps,
                "spread_pct": spread_pct,
                "turnover_proxy": parse_float(raw_row.get("turnover_proxy")),
                "actual_cost_bps": actual_cost_bps,
                "current_cost_hat_bps": current_cost_hat_bps,
                "cost_gap_bps": cost_gap_bps,
            }
        )

    prepared.sort(key=lambda row: row["event_time"])
    return prepared, since


def average(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def median_value(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = sorted(value for value in values if value is not None)
    if not valid:
        return None
    return float(median(valid))


def aggregate_metrics(rows: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "count": len(rows),
        "avg_actual_cost_bps": average(row.get("actual_cost_bps") for row in rows),
        "avg_current_cost_hat_bps": average(row.get("current_cost_hat_bps") for row in rows),
        "avg_cost_gap_bps": average(row.get("cost_gap_bps") for row in rows),
        "median_cost_gap_bps": median_value(row.get("cost_gap_bps") for row in rows),
        "avg_slippage_bps": average(row.get("slippage_bps") for row in rows),
        "avg_spread_pct": average(row.get("spread_pct") for row in rows),
        "avg_turnover_proxy": average(row.get("turnover_proxy") for row in rows),
    }


def summarize_overall(
    rows: List[Dict[str, object]],
    input_path: str,
    input_status: str,
    days: int,
    now: datetime,
    since: datetime,
    base_fee_bps: float,
    slippage_factor: float,
    close_timing_penalty_bps: float,
) -> List[Dict[str, object]]:
    metrics = aggregate_metrics(rows)
    return [
        {"metric": "generated_at", "value": now.strftime("%Y-%m-%d %H:%M:%S")},
        {"metric": "input_path", "value": input_path},
        {"metric": "input_status", "value": input_status},
        {"metric": "lookback_days", "value": normalize_days(days)},
        {"metric": "base_fee_bps", "value": base_fee_bps},
        {"metric": "slippage_factor", "value": slippage_factor},
        {"metric": "close_timing_penalty_bps", "value": close_timing_penalty_bps},
        {"metric": "target_period_start", "value": since.strftime("%Y-%m-%d %H:%M:%S")},
        {"metric": "target_period_end", "value": now.strftime("%Y-%m-%d %H:%M:%S")},
        {"metric": "eligible_rows", "value": metrics["count"]},
        {"metric": "avg_actual_cost_bps", "value": metrics["avg_actual_cost_bps"]},
        {"metric": "avg_current_cost_hat_bps", "value": metrics["avg_current_cost_hat_bps"]},
        {"metric": "avg_cost_gap_bps", "value": metrics["avg_cost_gap_bps"]},
        {"metric": "median_cost_gap_bps", "value": metrics["median_cost_gap_bps"]},
        {"metric": "avg_slippage_bps", "value": metrics["avg_slippage_bps"]},
        {"metric": "avg_spread_pct", "value": metrics["avg_spread_pct"]},
        {"metric": "avg_turnover_proxy", "value": metrics["avg_turnover_proxy"]},
    ]


def normalize_group_value(value: object, default: str = "unknown") -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    text = str(value or "").strip()
    return text or default


def group_by(rows: List[Dict[str, object]], group_col: str) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[normalize_group_value(row.get(group_col))].append(row)

    result: List[Dict[str, object]] = []
    for group_value, group_rows in grouped.items():
        aggregated = aggregate_metrics(group_rows)
        result.append({group_col: group_value, **aggregated})

    result.sort(key=lambda row: (-int(row["count"]), row[group_col]))
    return result


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary_text(summary_rows: List[Dict[str, object]], grouped_rows: Dict[str, List[Dict[str, object]]]) -> str:
    summary_map = {row["metric"]: row["value"] for row in summary_rows}
    lines = [
        "Cost Hat Gap Report",
        f"generated_at: {summary_map.get('generated_at', '')}",
        f"input_path: {summary_map.get('input_path', '')}",
        f"input_status: {summary_map.get('input_status', '')}",
        f"target_period: {summary_map.get('target_period_start', '')} -> {summary_map.get('target_period_end', '')}",
        f"lookback_days: {summary_map.get('lookback_days', '')}",
        f"base_fee_bps: {summary_map.get('base_fee_bps', '')}",
        f"slippage_factor: {summary_map.get('slippage_factor', '')}",
        f"close_timing_penalty_bps: {summary_map.get('close_timing_penalty_bps', '')}",
        f"eligible_rows: {summary_map.get('eligible_rows', 0)}",
        f"avg_actual_cost_bps: {summary_map.get('avg_actual_cost_bps', '')}",
        f"avg_current_cost_hat_bps: {summary_map.get('avg_current_cost_hat_bps', '')}",
        f"avg_cost_gap_bps: {summary_map.get('avg_cost_gap_bps', '')}",
        f"median_cost_gap_bps: {summary_map.get('median_cost_gap_bps', '')}",
    ]

    for key, title in (
        ("time_bucket", "top_time_buckets"),
        ("entry_or_exit", "entry_exit_breakdown"),
        ("is_force_exit", "force_exit_breakdown"),
        ("symbol", "top_symbols"),
    ):
        lines.append(f"{title}:")
        rows = grouped_rows.get(key, [])
        if not rows:
            lines.append("  - no data")
            continue
        for row in rows[:5]:
            lines.append(
                f"  - {row[key]}: count={row['count']}, avg_gap_bps={row['avg_cost_gap_bps']}, avg_hat_bps={row['avg_current_cost_hat_bps']}, avg_actual_bps={row['avg_actual_cost_bps']}"
            )
    return "\n".join(lines)


def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def main(
    env: str,
    days: int,
    input_path: str,
    out_dir: str,
    base_fee_bps: float,
    slippage_factor: float,
    close_timing_penalty_bps: float,
) -> int:
    resolved_input = input_path or get_default_input_path(env)
    ensure_dir(out_dir)

    raw_rows, input_status = read_csv_safe(resolved_input)
    now = datetime.now()
    prepared_rows, since = prepare_rows(
        raw_rows,
        days,
        now,
        base_fee_bps,
        slippage_factor,
        close_timing_penalty_bps,
    )

    summary_rows = summarize_overall(
        prepared_rows,
        resolved_input,
        input_status,
        days,
        now,
        since,
        base_fee_bps,
        slippage_factor,
        close_timing_penalty_bps,
    )
    by_time_bucket = group_by(prepared_rows, "time_bucket")
    by_symbol = group_by(prepared_rows, "symbol")
    by_entry_exit = group_by(prepared_rows, "entry_or_exit")
    by_force_exit = group_by(prepared_rows, "is_force_exit")
    grouped_rows = {
        "time_bucket": by_time_bucket,
        "symbol": by_symbol,
        "entry_or_exit": by_entry_exit,
        "is_force_exit": by_force_exit,
    }

    write_csv(os.path.join(out_dir, SUMMARY_PATH), summary_rows, SUMMARY_COLUMNS)
    write_csv(os.path.join(out_dir, BY_TIME_BUCKET_PATH), by_time_bucket, ["time_bucket"] + GROUP_METRIC_COLUMNS)
    write_csv(os.path.join(out_dir, BY_SYMBOL_PATH), by_symbol, ["symbol"] + GROUP_METRIC_COLUMNS)
    write_csv(os.path.join(out_dir, BY_ENTRY_EXIT_PATH), by_entry_exit, ["entry_or_exit"] + GROUP_METRIC_COLUMNS)
    write_csv(os.path.join(out_dir, BY_FORCE_EXIT_PATH), by_force_exit, ["is_force_exit"] + GROUP_METRIC_COLUMNS)

    summary_text = build_summary_text(summary_rows, grouped_rows)
    print(summary_text)
    write_text(os.path.join(out_dir, SUMMARY_TEXT_PATH), summary_text)

    if input_status in {"missing", "empty"}:
        print(f"input_status={input_status}: {resolved_input}")
    return 0


def cli() -> None:
    args = parse_args()
    raise SystemExit(
        main(
            args.env,
            args.days,
            args.input,
            args.out_dir,
            args.base_fee_bps,
            args.slippage_factor,
            args.close_timing_penalty_bps,
        )
    )


if __name__ == "__main__":
    cli()
