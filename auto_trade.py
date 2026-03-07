import asyncio
import argparse
import copy
import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except ModuleNotFoundError:
    aiohttp = None

from settings_loader import as_bool, as_float, as_int, load_settings, resolve_api_password

# =========================================================
# kabuステーション 自動取引エンジン (V2.1 完全突合・品質向上版)
# =========================================================

os.makedirs("logs", exist_ok=True)
log_filename = f"logs/auto_trade_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

EXEC_LOG_HEADER = [
    "order_id",
    "execution_id",
    "order_sent_time",
    "fill_time",
    "execution_order",
    "symbol",
    "side",
    "expected_ask",
    "expected_bid",
    "actual_price",
    "qty",
    "spread_pct",
    "slippage_yen",
]

ORDER_STATUS_HEADER = [
    "order_id",
    "symbol",
    "side",
    "expected_ask",
    "expected_bid",
    "order_sent_time",
    "status",
    "reason",
]

EXEC_LOG_PATH = "trade_execution_log.csv"
EXEC_LOG_PATH_SIM = "trade_execution_log_SIM.csv"
ORDER_STATUS_LOG_PATH = "order_status_log.csv"
ORDER_STATUS_LOG_PATH_SIM = "order_status_log_SIM.csv"
INITIAL_EQUITY = 1_000_000
PLACEHOLDER_PREFIXES = ("YOUR_",)
PLACEHOLDER_VALUES = {"", "dummy", "changeme", "replace_me", "your_password"}
HALT_FORCE_EXIT_UNRESOLVED = "HALT_FORCE_EXIT_UNRESOLVED"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pick_first_value(data: Optional[dict], keys: List[str]) -> Any:
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _safe_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _normalize_order_side(value: Any) -> str:
    raw_side = _safe_str(value).upper()
    if raw_side in {"1", "SELL", "S"}:
        return "SELL"
    if raw_side in {"2", "BUY", "B"}:
        return "BUY"
    return raw_side or "UNKNOWN"


def _extract_order_avg_price(order: dict) -> float:
    avg_price = safe_float(
        _pick_first_value(
            order,
            ["AvgPrice", "AveragePrice", "CumPrice", "ExecutionPrice"],
        ),
        0.0,
    )
    if avg_price > 0:
        return avg_price

    details = order.get("Details") or []
    if not isinstance(details, list):
        return 0.0

    total_value = 0.0
    total_qty = 0
    for detail in details:
        if not isinstance(detail, dict):
            continue
        price = safe_float(_pick_first_value(detail, ["Price", "ExecutionPrice"]), 0.0)
        qty = safe_int(_pick_first_value(detail, ["Qty", "CumQty", "ExecutionQty"]), 0)
        if price <= 0 or qty <= 0:
            continue
        total_value += price * qty
        total_qty += qty

    if total_qty <= 0:
        return 0.0
    return total_value / total_qty


def _normalize_order_state(order: dict, qty: int, cum_qty: int, remaining_qty: int) -> str:
    state_value = _pick_first_value(order, ["State", "OrderState", "Status", "OrderStatus"])
    state_code = safe_int(state_value, -1)
    state_text_parts = [
        _safe_str(_pick_first_value(order, ["StateName", "OrderStateName", "StatusText"])),
        _safe_str(_pick_first_value(order, ["Result", "OrderResult", "Message"])),
    ]
    state_text = " ".join(part.lower() for part in state_text_parts if part)

    if "reject" in state_text or "rejected" in state_text or "失敗" in state_text or "エラー" in state_text:
        return "REJECTED"
    if "expire" in state_text or "expired" in state_text or "失効" in state_text:
        return "EXPIRED"
    if "cancel" in state_text or "cancelled" in state_text or "canceled" in state_text or "取消" in state_text:
        return "CANCELED"

    if qty > 0 and cum_qty >= qty:
        return "FILLED"
    if cum_qty > 0 and remaining_qty > 0:
        return "PARTIAL"

    if state_code in {1, 2, 3, 4}:
        return "ORDERED"
    if state_code == 5:
        if qty > 0 and cum_qty >= qty:
            return "FILLED"
        if cum_qty > 0:
            return "PARTIAL"
        return "UNKNOWN"
    if state_code in {6}:
        return "CANCELED"
    if state_code in {7}:
        return "EXPIRED"
    if state_code in {8, 9}:
        return "REJECTED"

    return "UNKNOWN"


def _normalize_order_snapshot(order: dict) -> Optional[dict]:
    if not isinstance(order, dict):
        return None

    order_id = _safe_str(_pick_first_value(order, ["ID", "OrderID", "OrderId"]))
    if not order_id:
        return None

    qty = safe_int(_pick_first_value(order, ["OrderQty", "Qty"]), 0)
    cum_qty = safe_int(_pick_first_value(order, ["CumQty", "ExecutionQty", "FilledQty"]), 0)
    remaining_qty_value = _pick_first_value(order, ["LeavesQty", "RemainingQty"])
    remaining_qty = safe_int(remaining_qty_value, max(qty - cum_qty, 0))
    if remaining_qty < 0:
        remaining_qty = 0

    snapshot = {
        "order_id": order_id,
        "symbol": _safe_str(_pick_first_value(order, ["Symbol", "Ticker"])),
        "side": _normalize_order_side(_pick_first_value(order, ["Side"])),
        "state": _normalize_order_state(order, qty, cum_qty, remaining_qty),
        "qty": qty,
        "cum_qty": cum_qty,
        "remaining_qty": remaining_qty,
        "price": safe_float(_pick_first_value(order, ["Price", "OrderPrice"]), 0.0),
        "avg_price": _extract_order_avg_price(order),
        "sent_at": _safe_str(
            _pick_first_value(
                order,
                ["RecvTime", "OrderTime", "SendTime", "TransactTime"],
            )
        ),
        "raw": copy.deepcopy(order),
    }
    return snapshot


async def fetch_orders_snapshot(api: "KabuAPI") -> Dict[str, dict]:
    response = await api.get_orders()
    if not response:
        return {}

    if isinstance(response, list):
        orders = response
    elif isinstance(response, dict):
        nested_orders = _pick_first_value(response, ["Orders", "orders", "data"])
        if isinstance(nested_orders, list):
            orders = nested_orders
        else:
            orders = [response]
    else:
        return {}

    snapshots: Dict[str, dict] = {}
    for order in orders:
        snapshot = _normalize_order_snapshot(order)
        if snapshot is None:
            continue
        snapshots[snapshot["order_id"]] = snapshot
    return snapshots


def _normalize_position_state(position: dict, qty: int, leaves_qty: int) -> str:
    state_value = _pick_first_value(position, ["State", "PositionState", "Status", "PositionStatus"])
    state_code = safe_int(state_value, -1)
    state_text_parts = [
        _safe_str(_pick_first_value(position, ["StateName", "PositionStateName", "StatusText"])),
        _safe_str(_pick_first_value(position, ["Message", "Result"])),
    ]
    state_text = " ".join(part.lower() for part in state_text_parts if part)

    if "open" in state_text or "active" in state_text or "保有" in state_text or "建玉" in state_text:
        return "OPEN"
    if "close" in state_text or "closed" in state_text or "決済" in state_text:
        return "CLOSED"

    if qty > 0 or leaves_qty > 0:
        return "OPEN"
    if state_code == 0:
        return "UNKNOWN"
    if state_code > 0:
        return "CLOSED"
    return "UNKNOWN"


def _normalize_position_snapshot(position: dict) -> Optional[dict]:
    if not isinstance(position, dict):
        return None

    hold_id = _safe_str(_pick_first_value(position, ["HoldID", "ExecutionID"]))
    if not hold_id:
        return None

    hold_qty = safe_int(_pick_first_value(position, ["HoldQty", "Qty"]), 0)
    leaves_qty = safe_int(_pick_first_value(position, ["LeavesQty", "RemainingQty"]), 0)
    qty = hold_qty if hold_qty > 0 else leaves_qty

    snapshot = {
        "hold_id": hold_id,
        "symbol": _safe_str(_pick_first_value(position, ["Symbol", "Ticker"])),
        "side": _normalize_order_side(_pick_first_value(position, ["Side"])),
        "qty": qty,
        "price": safe_float(_pick_first_value(position, ["Price", "ExecutionPrice", "EntryPrice"]), 0.0),
        "leaves_qty": leaves_qty,
        "state": _normalize_position_state(position, qty, leaves_qty),
        "raw": copy.deepcopy(position),
    }
    return snapshot


async def fetch_positions_snapshot(api: "KabuAPI", product: int) -> Dict[str, dict]:
    response = await api.get_positions(product=product)
    if not response:
        return {}

    if isinstance(response, list):
        positions = response
    elif isinstance(response, dict):
        nested_positions = _pick_first_value(response, ["Positions", "positions", "data"])
        if isinstance(nested_positions, list):
            positions = nested_positions
        else:
            positions = [response]
    else:
        return {}

    snapshots: Dict[str, dict] = {}
    for position in positions:
        snapshot = _normalize_position_snapshot(position)
        if snapshot is None:
            continue
        snapshots[snapshot["hold_id"]] = snapshot
    return snapshots


def _coerce_state_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _opposite_trade_side(side: str) -> str:
    normalized = _normalize_order_side(side)
    if normalized == "BUY":
        return "SELL"
    if normalized == "SELL":
        return "BUY"
    return "UNKNOWN"


def _extract_local_trade_context(local_state: Any) -> dict:
    state = _coerce_state_dict(local_state)
    order_side = _normalize_order_side(
        _pick_first_value(state, ["order_side", "pending_side", "exit_side", "entry_side"])
    )
    position_side = _normalize_order_side(_pick_first_value(state, ["position_side", "side", "Side"]))
    symbol = _safe_str(_pick_first_value(state, ["symbol", "Symbol", "ticker", "Ticker"]))
    hold_id = _safe_str(_pick_first_value(state, ["hold_id", "HoldID", "execution_id", "ExecutionID"]))
    order_id = _safe_str(
        _pick_first_value(
            state,
            ["order_id", "OrderID", "OrderId", "exit_order_id", "entry_order_id"],
        )
    )
    qty = safe_int(
        _pick_first_value(state, ["qty", "Qty", "position_qty", "target_qty", "order_qty", "remaining_qty"]),
        0,
    )
    is_closing = as_bool(
        _pick_first_value(state, ["is_closing", "closing", "close_requested", "exit_pending", "is_exit"]),
        False,
    )
    phase_markers = [
        _safe_str(_pick_first_value(state, ["phase", "intent", "action"])).upper(),
        _safe_str(_pick_first_value(state, ["status", "trade_status", "state"])).upper(),
    ]
    phase_text = " ".join(marker for marker in phase_markers if marker)

    phase = ""
    if is_closing or any(token in phase_text for token in {"EXIT", "CLOSE", "CLOSING", "SELL_EXIT"}):
        phase = "EXIT"
    elif any(token in phase_text for token in {"ENTRY", "OPEN", "BUY_ENTRY"}):
        phase = "ENTRY"
    elif order_side == "SELL" and hold_id:
        phase = "EXIT"
    elif order_side == "BUY":
        phase = "ENTRY"

    return {
        "symbol": symbol,
        "hold_id": hold_id,
        "order_id": order_id,
        "qty": qty,
        "phase": phase,
        "order_side": order_side,
        "position_side": position_side,
        "is_closing": is_closing,
        "raw": state,
    }


def _match_position_snapshot(local_context: dict, positions_snapshot: Dict[str, dict]) -> Tuple[Optional[dict], List[str]]:
    notes: List[str] = []
    hold_id = local_context.get("hold_id", "")
    symbol = local_context.get("symbol", "")
    qty = safe_int(local_context.get("qty"), 0)
    position_side = local_context.get("position_side", "UNKNOWN")

    if hold_id:
        matched = positions_snapshot.get(hold_id)
        if matched:
            notes.append("matched position by hold_id")
            return matched, notes
        notes.append("hold_id not found in positions snapshot")

    if not symbol:
        if len(positions_snapshot) == 1:
            notes.append("matched sole position snapshot without local hint")
            return next(iter(positions_snapshot.values())), notes
        return None, notes

    candidates = [position for position in positions_snapshot.values() if position.get("symbol") == symbol]
    if position_side not in {"", "UNKNOWN"}:
        side_filtered = [position for position in candidates if position.get("side") == position_side]
        if side_filtered:
            candidates = side_filtered
    if qty > 0:
        qty_filtered = [position for position in candidates if safe_int(position.get("qty"), 0) == qty]
        if qty_filtered:
            candidates = qty_filtered
    if len(candidates) == 1:
        notes.append("matched position by symbol")
        return candidates[0], notes
    if len(candidates) > 1:
        notes.append("multiple positions matched symbol; selected first candidate")
        return candidates[0], notes
    return None, notes


def _select_preferred_order(candidates: List[dict]) -> Optional[dict]:
    if not candidates:
        return None
    priority = {
        "PARTIAL": 0,
        "ORDERED": 1,
        "FILLED": 2,
        "CANCELED": 3,
        "EXPIRED": 4,
        "REJECTED": 5,
        "UNKNOWN": 6,
    }
    return sorted(
        candidates,
        key=lambda order: (
            priority.get(_safe_str(order.get("state")), 99),
            _safe_str(order.get("sent_at")),
            _safe_str(order.get("order_id")),
        ),
        reverse=False,
    )[0]


def _match_order_snapshot(
    local_context: dict,
    orders_snapshot: Dict[str, dict],
    matched_position: Optional[dict],
) -> Tuple[Optional[dict], List[str]]:
    notes: List[str] = []
    order_id = local_context.get("order_id", "")
    symbol = local_context.get("symbol", "")
    phase = local_context.get("phase", "")
    order_side = local_context.get("order_side", "UNKNOWN")
    position_side = local_context.get("position_side", "UNKNOWN")

    if order_id:
        matched = orders_snapshot.get(order_id)
        if matched:
            notes.append("matched order by order_id")
            return matched, notes
        notes.append("order_id not found in orders snapshot")

    if not symbol:
        if len(orders_snapshot) == 1:
            notes.append("matched sole order snapshot without local hint")
            return next(iter(orders_snapshot.values())), notes
        return None, notes

    expected_side = order_side
    if expected_side in {"", "UNKNOWN"}:
        if phase == "EXIT":
            position_side = matched_position.get("side") if matched_position else position_side
            expected_side = _opposite_trade_side(position_side)
        elif phase == "ENTRY" and position_side not in {"", "UNKNOWN"}:
            expected_side = position_side

    candidates = [order for order in orders_snapshot.values() if order.get("symbol") == symbol]
    if expected_side not in {"", "UNKNOWN"}:
        side_filtered = [order for order in candidates if order.get("side") == expected_side]
        if side_filtered:
            candidates = side_filtered

    matched = _select_preferred_order(candidates)
    if matched:
        notes.append("matched order by symbol")
    return matched, notes


def _resolve_status_from_order_only(order: dict) -> str:
    state = _safe_str(order.get("state"))
    if state == "PARTIAL":
        return "PARTIAL"
    if state == "ORDERED":
        return "PENDING"
    return "UNKNOWN"


def reconcile_trade_state(
    local_state: Any,
    orders_snapshot: Optional[Dict[str, dict]],
    positions_snapshot: Optional[Dict[str, dict]],
) -> dict:
    orders_snapshot = orders_snapshot if isinstance(orders_snapshot, dict) else {}
    positions_snapshot = positions_snapshot if isinstance(positions_snapshot, dict) else {}
    local_context = _extract_local_trade_context(local_state)
    notes: List[str] = []

    matched_position, position_notes = _match_position_snapshot(local_context, positions_snapshot)
    matched_order, order_notes = _match_order_snapshot(local_context, orders_snapshot, matched_position)
    notes.extend(position_notes)
    notes.extend(order_notes)

    phase = local_context.get("phase", "")
    local_qty = safe_int(local_context.get("qty"), 0)
    position_qty = safe_int((matched_position or {}).get("qty"), 0)
    position_leaves_qty = safe_int((matched_position or {}).get("leaves_qty"), position_qty)
    order_remaining_qty = safe_int((matched_order or {}).get("remaining_qty"), 0)
    order_cum_qty = safe_int((matched_order or {}).get("cum_qty"), 0)
    order_state = _safe_str((matched_order or {}).get("state"))

    response = {
        "result": "UNKNOWN",
        "resolved_status": "UNKNOWN",
        "matched_order": matched_order,
        "matched_position": matched_position,
        "remaining_qty": 0,
        "notes": notes,
    }

    if phase == "EXIT":
        if matched_position is None:
            response["result"] = "MATCHED_CLOSED"
            response["resolved_status"] = "CLOSED"
            response["remaining_qty"] = 0
            notes.append("positions snapshot has no matching hold_id/symbol; treated as closed")
            if matched_order:
                notes.append("positions truth takes precedence over exit order state")
            return response

        response["resolved_status"] = "OPEN"
        response["remaining_qty"] = position_leaves_qty if position_leaves_qty > 0 else position_qty
        if matched_order is None:
            response["result"] = "EXIT_UNRESOLVED"
            notes.append("position remains open but no matching exit order was found")
            return response

        if order_state == "PARTIAL" or (local_qty > 0 and response["remaining_qty"] < local_qty) or order_cum_qty > 0:
            response["result"] = "EXIT_PARTIAL"
            notes.append("exit order is partially filled while position still remains")
            return response

        response["result"] = "EXIT_PENDING"
        notes.append("exit order exists but position still remains open")
        return response

    if matched_position is not None:
        response["resolved_status"] = "OPEN"
        response["remaining_qty"] = position_leaves_qty if position_leaves_qty > 0 else position_qty
        if phase == "ENTRY":
            if matched_order and (
                order_state == "PARTIAL" or (local_qty > 0 and position_qty < local_qty) or order_remaining_qty > 0
            ):
                response["result"] = "ENTRY_PARTIAL"
                notes.append("entry position exists and order still has remaining quantity")
                return response
            response["result"] = "MATCHED_OPEN"
            notes.append("positions snapshot confirms open position")
            return response

        response["result"] = "POSITION_ONLY"
        notes.append("position exists without a conclusive entry/exit phase")
        return response

    if matched_order is not None:
        response["remaining_qty"] = order_remaining_qty
        response["resolved_status"] = _resolve_status_from_order_only(matched_order)
        if phase == "ENTRY":
            if order_state == "PARTIAL":
                response["result"] = "ENTRY_PARTIAL"
                notes.append("entry order is partially filled and no position is visible yet")
                return response
            if order_state == "ORDERED":
                response["result"] = "ENTRY_PENDING"
                notes.append("entry order is waiting and no position is visible yet")
                return response
        response["result"] = "ORDER_ONLY"
        notes.append("order exists without a matching position")
        return response

    if phase == "ENTRY":
        response["remaining_qty"] = local_qty
        notes.append("entry intent exists but neither position nor order matched")
        return response

    notes.append("no matching position or order was found")
    return response


def _extract_position_context(position: Any, local_state: Any) -> dict:
    position_data = _coerce_state_dict(position)
    local_data = _coerce_state_dict(local_state)
    symbol = _safe_str(_pick_first_value(position_data, ["symbol", "Symbol", "ticker", "Ticker"]))
    hold_id = _safe_str(
        _pick_first_value(
            position_data,
            ["hold_id", "HoldID", "execution_id", "ExecutionID"],
        )
    ) or _safe_str(_pick_first_value(local_data, ["hold_id", "HoldID", "execution_id", "ExecutionID"]))
    qty = safe_int(_pick_first_value(position_data, ["qty", "Qty", "position_qty"]), 0)
    exchange = as_int(_pick_first_value(position_data, ["exchange", "Exchange"]), 1)
    position_side = _normalize_order_side(
        _pick_first_value(
            position_data,
            ["side", "Side", "position_side"],
        )
        or _pick_first_value(local_data, ["position_side", "side", "Side"])
        or "BUY"
    )
    exit_side = _opposite_trade_side(position_side)
    if exit_side == "UNKNOWN":
        exit_side = "SELL"

    return {
        "symbol": symbol,
        "hold_id": hold_id,
        "qty": qty,
        "exchange": exchange,
        "position_side": position_side,
        "exit_side": exit_side,
    }


async def request_exit(api: "KabuAPI", position: Any, local_state: Any, reason: str) -> dict:
    local_data = _coerce_state_dict(local_state)
    position_context = _extract_position_context(position, local_state)
    requested_at = now_str()
    current_attempt = safe_int(_pick_first_value(local_data, ["exit_attempt_no"]), 0)
    next_attempt = current_attempt + 1

    response = {
        "ok": False,
        "exit_order_id": "",
        "attempt_no": next_attempt,
        "requested_at": requested_at,
        "reason": reason,
        "error": "",
    }

    symbol = position_context["symbol"]
    hold_id = position_context["hold_id"]
    qty = position_context["qty"]
    exchange = position_context["exchange"]
    exit_side = "1" if position_context["exit_side"] == "SELL" else "2"

    local_data["exit_attempt_no"] = next_attempt
    local_data["last_exit_reason"] = reason
    local_data["last_exit_attempt_at"] = requested_at
    if hold_id:
        local_data["hold_id"] = hold_id
    if qty > 0:
        local_data["known_position_qty"] = qty

    if not hasattr(api, "send_order"):
        response["error"] = "api.send_order is not available"
        local_data["last_exit_error"] = response["error"]
        return response
    if not symbol:
        response["error"] = "missing symbol for exit request"
        local_data["last_exit_error"] = response["error"]
        return response
    if qty <= 0:
        response["error"] = "missing qty for exit request"
        local_data["last_exit_error"] = response["error"]
        return response
    if not hold_id:
        response["error"] = "missing hold_id for exit request"
        local_data["last_exit_error"] = response["error"]
        return response

    try:
        send_order = getattr(api, "send_order")
        result = send_order(
            symbol,
            side=exit_side,
            qty=qty,
            is_close=True,
            hold_id=hold_id,
            exchange=exchange,
        )
        if asyncio.iscoroutine(result):
            result = await result
    except Exception as exc:
        response["error"] = str(exc)
        local_data["last_exit_error"] = response["error"]
        logger.error(f"❌ 決済注文送信例外: {symbol} reason={reason} error={exc}")
        return response

    if not isinstance(result, dict) or result.get("Result") != 0:
        response["error"] = _safe_str(_pick_first_value(result or {}, ["Message", "ErrorMessage", "Code"])) or str(result)
        local_data["last_exit_error"] = response["error"]
        logger.error(f"❌ 決済注文送信失敗: {symbol} reason={reason} result={result}")
        return response

    exit_order_id = _safe_str(_pick_first_value(result, ["OrderId", "OrderID", "ID"]))
    if not exit_order_id:
        response["error"] = "exit order accepted without order id"
        local_data["last_exit_error"] = response["error"]
        logger.error(f"❌ 決済注文送信失敗: {symbol} reason={reason} result={result}")
        return response

    local_data["exit_order_id"] = exit_order_id
    local_data["exit_requested_at"] = requested_at
    local_data["status"] = "EXIT_SENT"
    local_data["last_exit_error"] = ""

    response["ok"] = True
    response["exit_order_id"] = exit_order_id
    response["error"] = ""
    logger.info(f"✅ 決済注文送信受付: {symbol} hold_id={hold_id} order_id={exit_order_id} reason={reason}")
    return response


def _parse_state_timestamp(value: Any) -> Optional[datetime]:
    text = _safe_str(value)
    if not text:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _resolve_exit_retry_limit(local_state: Any) -> int:
    local_data = _coerce_state_dict(local_state)
    retry_limit = safe_int(
        _pick_first_value(local_data, ["exit_retry_limit", "max_exit_attempts", "exit_max_attempts"]),
        3,
    )
    return max(retry_limit, 1)


def confirm_exit(
    local_state: Any,
    orders_snapshot: Optional[Dict[str, dict]],
    positions_snapshot: Optional[Dict[str, dict]],
    ttl_sec: int = 30,
) -> dict:
    local_data = _coerce_state_dict(local_state)
    reconciliation = reconcile_trade_state(local_state, orders_snapshot, positions_snapshot)
    matched_position = reconciliation.get("matched_position") or {}
    remaining_qty = safe_int(reconciliation.get("remaining_qty"), 0)
    if remaining_qty <= 0 and matched_position:
        remaining_qty = safe_int(
            _pick_first_value(matched_position, ["leaves_qty", "qty"]),
            0,
        )

    known_position_qty = safe_int(
        _pick_first_value(local_data, ["known_position_qty", "qty", "position_qty"]),
        0,
    )
    exit_attempt_no = safe_int(_pick_first_value(local_data, ["exit_attempt_no"]), 0)
    retry_limit = _resolve_exit_retry_limit(local_state)
    attempted_at = _parse_state_timestamp(
        _pick_first_value(local_data, ["exit_requested_at", "last_exit_attempt_at"])
    )
    ttl_expired = False
    if attempted_at is not None:
        ttl_expired = (datetime.now() - attempted_at).total_seconds() > max(ttl_sec, 0)

    if reconciliation.get("result") == "MATCHED_CLOSED":
        return {
            "result": "CLOSED",
            "remaining_qty": 0,
            "should_retry": False,
            "reason": "positions snapshot no longer contains the hold_id",
        }

    if remaining_qty > 0 and known_position_qty > 0 and remaining_qty < known_position_qty:
        return {
            "result": "PARTIAL",
            "remaining_qty": remaining_qty,
            "should_retry": False,
            "reason": "position remains but quantity decreased after exit request",
        }

    if reconciliation.get("result") == "EXIT_PARTIAL":
        return {
            "result": "PARTIAL",
            "remaining_qty": remaining_qty,
            "should_retry": False,
            "reason": "exit order is partially filled",
        }

    if not ttl_expired:
        return {
            "result": "PENDING",
            "remaining_qty": remaining_qty,
            "should_retry": False,
            "reason": "exit confirmation is waiting within ttl",
        }

    if exit_attempt_no < retry_limit:
        return {
            "result": "RETRY",
            "remaining_qty": remaining_qty,
            "should_retry": True,
            "reason": "exit confirmation exceeded ttl and retry is still available",
        }

    return {
        "result": "UNRESOLVED",
        "remaining_qty": remaining_qty,
        "should_retry": False,
        "reason": "exit confirmation exceeded ttl and retry limit is exhausted",
    }


class Config:
    DEFAULT_CONFIG = {
        "IS_PRODUCTION": False,
        "API_PASSWORD": "",
        "API_PASSWORD_SIM": "YOUR_SIM_API_PASSWORD",
        "API_PASSWORD_PROD": "YOUR_PROD_API_PASSWORD",
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
        "STOP_LOSS_PCT": 0.05,
        "ATR_K1": 2.0,
        "ATR_K2": 3.0,
        "RECO_CSV_PATH": "recommendations.csv",
    }

    def __init__(self):
        self.config_file = "settings.yml"
        self.local_config_file = "settings.local.yml"
        self.load_config()

    def load_config(self) -> None:
        self.config_data = load_settings(
            self.DEFAULT_CONFIG,
            logger=logger,
            settings_path=self.config_file,
            local_settings_path=self.local_config_file,
        )

        self.IS_PRODUCTION = as_bool(
            self.config_data.get("IS_PRODUCTION"),
            self.DEFAULT_CONFIG["IS_PRODUCTION"],
        )
        self.PORT = 18080 if self.IS_PRODUCTION else 18081
        self.API_URL = f"http://localhost:{self.PORT}/kabusapi"
        self.API_PASSWORD = resolve_api_password(self.config_data, is_production=self.IS_PRODUCTION)
        self.TRADE_PASSWORD = str(
            self.config_data.get("TRADE_PASSWORD", self.DEFAULT_CONFIG["TRADE_PASSWORD"])
        )
        self.EXCHANGE = as_int(self.config_data.get("EXCHANGE"), self.DEFAULT_CONFIG["EXCHANGE"])
        self.TRADE_STYLE = str(
            self.config_data.get("TRADE_STYLE", self.DEFAULT_CONFIG["TRADE_STYLE"])
        ).lower()
        self.TARGET_HORIZON = str(
            self.config_data.get("TARGET_HORIZON", self.DEFAULT_CONFIG["TARGET_HORIZON"])
        )
        self.TRADE_MODE = str(
            self.config_data.get("TRADE_MODE", self.DEFAULT_CONFIG["TRADE_MODE"])
        ).upper()

        account_type_raw = self.config_data.get(
            "ACCOUNT_TYPE",
            self.config_data.get("FUND_TYPE", self.DEFAULT_CONFIG["ACCOUNT_TYPE"]),
        )
        self.ACCOUNT_TYPE = as_int(account_type_raw, self.DEFAULT_CONFIG["ACCOUNT_TYPE"])

        self.MAX_POSITIONS = as_int(
            self.config_data.get("MAX_POSITIONS"),
            self.DEFAULT_CONFIG["MAX_POSITIONS"],
        )
        self.LOT_CALC_MODE = str(
            self.config_data.get("LOT_CALC_MODE", self.DEFAULT_CONFIG["LOT_CALC_MODE"])
        ).upper()
        self.FIXED_LOT_SIZE = as_int(
            self.config_data.get("FIXED_LOT_SIZE"),
            self.DEFAULT_CONFIG["FIXED_LOT_SIZE"],
        )
        self.AUTO_INVEST_RATIO = as_float(
            self.config_data.get("AUTO_INVEST_RATIO"),
            self.DEFAULT_CONFIG["AUTO_INVEST_RATIO"],
        )
        self.ENTRY_THRESHOLD_PROB = as_float(
            self.config_data.get("ENTRY_THRESHOLD_PROB"),
            self.DEFAULT_CONFIG["ENTRY_THRESHOLD_PROB"],
        )
        self.TAKE_PROFIT_PCT = as_float(
            self.config_data.get("TAKE_PROFIT_PCT"),
            self.DEFAULT_CONFIG["TAKE_PROFIT_PCT"],
        )
        self.STOP_LOSS_PCT = as_float(
            self.config_data.get("STOP_LOSS_PCT"),
            self.DEFAULT_CONFIG["STOP_LOSS_PCT"],
        )
        self.ATR_K1 = as_float(self.config_data.get("ATR_K1"), self.DEFAULT_CONFIG["ATR_K1"])
        self.ATR_K2 = as_float(self.config_data.get("ATR_K2"), self.DEFAULT_CONFIG["ATR_K2"])
        self.RECO_CSV_PATH = str(
            self.config_data.get("RECO_CSV_PATH", self.DEFAULT_CONFIG["RECO_CSV_PATH"])
        )


class TokenBucket:
    def __init__(self, capacity: int, fill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.fill_rate = fill_rate
        self.last_fill = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, amount: int = 1) -> None:
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


class KabuAPI:
    def __init__(self, config: Config):
        self.config = config
        self.token: Optional[str] = None
        self.session = None
        self.bucket = TokenBucket(capacity=5, fill_rate=5.0)

    async def start_session(self) -> None:
        if aiohttp is None:
            raise RuntimeError("aiohttp is not installed. Install dependencies before normal execution.")
        self.session = aiohttp.ClientSession()

    async def close_session(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> Optional[dict]:
        if self.session is None:
            logger.error("API session has not been started")
            return None

        await self.bucket.consume()
        url = f"{self.config.API_URL}/{endpoint}"
        headers = {"X-API-KEY": self.token} if self.token else {}

        try:
            async with self.session.request(
                method,
                url,
                headers=headers,
                json=data,
                params=params,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API HTTP Error {response.status} at {endpoint}: {error_text}")
                response.raise_for_status()
                return await response.json()
        except Exception as exc:
            logger.error(f"API Request Exception ({endpoint}): {exc}")
            return None

    async def get_token(self) -> None:
        data = {"APIPassword": self.config.API_PASSWORD}
        res = await self._request("POST", "token", data=data)
        if res and "Token" in res:
            self.token = res["Token"]
            logger.info("✅ トークン取得成功")
        else:
            logger.error("❌ トークン取得失敗")

    async def get_board(self, symbol: str, exchange: int):
        target_exchange = 1 if as_int(exchange, 1) == 9 else as_int(exchange, 1)
        return await self._request("GET", f"board/{symbol}@{target_exchange}")

    async def get_wallet_cash(self):
        return await self._request("GET", "wallet/cash")

    async def get_positions(self, product: int = 0):
        return await self._request("GET", "positions", params={"product": product})

    async def get_orders(self, product: int = 0):
        params = {"product": product} if product != 0 else {}
        return await self._request("GET", "orders", params=params)

    async def send_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float = 0,
        is_close: bool = False,
        hold_id: Optional[str] = None,
        exchange: Optional[int] = None,
    ):
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
            "FrontOrderType": 10,
        }

        if self.config.TRADE_MODE == "MARGIN" and is_close:
            if hold_id:
                order_data["ClosePositions"] = [{"HoldID": hold_id, "Qty": int(qty)}]
            else:
                logger.warning("⚠️ 信用返済ですが建玉ID(HoldID)が指定されていません")

        action = "買" if side == "2" else "売"
        trade_type_str = "現物" if self.config.TRADE_MODE == "CASH" else ("信用返済" if is_close else "信用新規")

        if not self.config.IS_PRODUCTION:
            logger.info(
                f"🧪 [シミュレーター] 発注スキップ [{trade_type_str}]: "
                f"{action} {symbol} {qty}株 (成行) [市場: {target_exchange}]"
            )
            return {"Result": 0, "OrderId": f"SIM_{int(time.time())}"}

        logger.info(
            f"🚀 発注リクエスト送信 [{trade_type_str}]: "
            f"{action} {symbol} {qty}株 (成行) [市場: {target_exchange}]"
        )
        return await self._request("POST", "sendorder", data=order_data)


class MarketData:
    def __init__(self, api: KabuAPI):
        self.api = api

    async def safe_get_board(self, symbol: str, exchange: int) -> Optional[dict]:
        board = await self.api.get_board(symbol, exchange)
        if board is None:
            return None
        if not isinstance(board, dict):
            logger.warning(f"⚠️ 板情報の形式が不正です: {symbol}")
            return None
        return board

    async def safe_get_price(self, symbol: str, exchange: int) -> Tuple[Optional[dict], Optional[float]]:
        board = await self.safe_get_board(symbol, exchange)
        return board, self.extract_reference_price(board)

    @staticmethod
    def extract_best_bid_ask(board: Optional[dict]) -> Tuple[float, float]:
        if not isinstance(board, dict):
            return 0.0, 0.0

        sell1 = board.get("Sell1")
        buy1 = board.get("Buy1")
        ask = 0.0
        bid = 0.0

        if isinstance(sell1, dict):
            ask = safe_float(sell1.get("Price"), 0.0)
        if ask <= 0:
            ask = safe_float(
                board.get("AskPrice")
                or board.get("AskPrice1")
                or board.get("Ask"),
                0.0,
            )

        if isinstance(buy1, dict):
            bid = safe_float(buy1.get("Price"), 0.0)
        if bid <= 0:
            bid = safe_float(
                board.get("BidPrice")
                or board.get("BidPrice1")
                or board.get("Bid"),
                0.0,
            )

        return ask, bid

    @staticmethod
    def extract_reference_price(board: Optional[dict]) -> Optional[float]:
        if not isinstance(board, dict):
            return None

        for key in ("CurrentPrice", "PreviousClose"):
            price = safe_float(board.get(key), 0.0)
            if price > 0:
                return price
        return None

    @staticmethod
    def extract_actual_price(api_order: Optional[dict]) -> Optional[float]:
        if not isinstance(api_order, dict):
            return None

        details = api_order.get("Details") or []
        total_value = 0.0
        total_qty = 0

        for detail in details:
            if safe_int(detail.get("State"), 0) != 5:
                continue
            price = safe_float(detail.get("Price"), 0.0)
            qty = safe_int(detail.get("Qty"), 0)
            if price <= 0 or qty <= 0:
                continue
            total_value += price * qty
            total_qty += qty

        if total_qty <= 0:
            return None
        return total_value / total_qty

    @staticmethod
    def calculate_spread_pct(ask: float, bid: float) -> float:
        return ((ask - bid) / bid) if bid > 0 else 0.0


class ExecutionLogger:
    def __init__(self, is_sim: bool):
        self.is_sim = is_sim
        self.exec_log_path = EXEC_LOG_PATH_SIM if is_sim else EXEC_LOG_PATH
        self.status_log_path = ORDER_STATUS_LOG_PATH_SIM if is_sim else ORDER_STATUS_LOG_PATH

    @staticmethod
    def _ensure_parent_dir(path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _write_row(self, path: str, header: List[str], row: dict) -> None:
        self._ensure_header(path, header)
        normalized = {field: row.get(field, "") for field in header}
        with open(path, "a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writerow(normalized)

    def _ensure_header(self, path: str, header: List[str]) -> None:
        self._ensure_parent_dir(path)
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        if file_exists:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)

    def log_trade_execution(self, row: dict) -> None:
        self._write_row(self.exec_log_path, EXEC_LOG_HEADER, row)

    def log_order_status(self, row: dict) -> None:
        self._write_row(self.status_log_path, ORDER_STATUS_HEADER, row)

    def ensure_log_headers(self) -> None:
        self._ensure_header(self.exec_log_path, EXEC_LOG_HEADER)
        self._ensure_header(self.status_log_path, ORDER_STATUS_HEADER)


@dataclass
class PendingOrder:
    order_id: str
    order_sent_time: str
    execution_order: int
    expected_ask: float
    expected_bid: float
    spread_pct: float
    time_added: float
    side: str


@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    stop_loss_price: float
    highest_price: float
    hold_id: str
    exchange: int
    is_closing: bool = False
    atr: float = 0.0


@dataclass
class TradingState:
    daily_loss_limit: float = INITIAL_EQUITY * -0.02
    today_realized_pnl: float = 0.0
    trading_halted: bool = False
    halt_reason: str = ""
    unresolved_exit_count: int = 0


class PortfolioManager:
    PENDING_TTL_SECONDS = 600

    def __init__(
        self,
        config: Config,
        api: KabuAPI,
        market_data: MarketData,
        execution_logger: ExecutionLogger,
        trading_state: TradingState,
    ):
        self.config = config
        self.api = api
        self.market_data = market_data
        self.execution_logger = execution_logger
        self.trading_state = trading_state
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, List[PendingOrder]] = {}
        self.seen_exec_ids = set()
        self.active_signals: Dict[str, float] = {}

    def set_active_signals(self, signals: List[dict]) -> None:
        self.active_signals = {
            signal["symbol"]: safe_float(signal.get("atr"), 0.0)
            for signal in signals
            if signal.get("symbol")
        }

    def register_pending(
        self,
        symbol: str,
        order_id: str,
        order_sent_time: str,
        execution_order: int,
        expected_ask: float,
        expected_bid: float,
        side: str,
    ) -> None:
        pending = PendingOrder(
            order_id=order_id,
            order_sent_time=order_sent_time,
            execution_order=execution_order,
            expected_ask=expected_ask,
            expected_bid=expected_bid,
            spread_pct=MarketData.calculate_spread_pct(expected_ask, expected_bid),
            time_added=time.time(),
            side=side,
        )
        self.pending_orders.setdefault(symbol, []).append(pending)

    def _find_best_pending(self, symbol: str, side: str) -> Optional[PendingOrder]:
        pending_list = self.pending_orders.get(symbol, [])
        side_pendings = [pending for pending in pending_list if pending.side == side]
        if not side_pendings:
            return None

        now = time.time()
        best_pending = min(side_pendings, key=lambda pending: abs(now - pending.time_added))
        remaining = [pending for pending in pending_list if pending is not best_pending]
        if remaining:
            self.pending_orders[symbol] = remaining
        else:
            self.pending_orders.pop(symbol, None)
        return best_pending

    def add_position(
        self,
        symbol: str,
        qty: int,
        entry_price: float,
        exchange: int,
        hold_id: str,
    ) -> None:
        atr = self.active_signals.get(symbol, 0.0)
        stop_pct = self.config.ATR_K1 * atr if atr > 0 else self.config.STOP_LOSS_PCT
        stop_loss_price = entry_price * (1 - stop_pct)
        self.positions[symbol] = Position(
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            highest_price=entry_price,
            hold_id=hold_id,
            exchange=exchange,
            atr=atr,
        )
        logger.info(
            f"💼 ポジション登録: {symbol} {qty}株 "
            f"(単価: {entry_price:,.1f}円 / 損切幅: {stop_pct * 100:.1f}%)"
        )

    async def fetch_active_orders(self, log: bool = True):
        active_count = 0
        active_symbols = []
        active_details = []
        orders = await self.api.get_orders()
        if not isinstance(orders, list):
            return active_count, active_symbols, active_details, orders or []

        for order in orders:
            state = safe_int(order.get("State"), 0)
            order_qty = safe_int(order.get("OrderQty"), 0)
            cum_qty = safe_int(order.get("CumQty"), 0)
            if state == 5 or cum_qty >= order_qty:
                continue

            symbol = order.get("Symbol")
            active_count += 1
            if symbol:
                active_symbols.append(symbol)
                active_details.append(
                    {
                        "symbol": symbol,
                        "state": state,
                        "order_qty": order_qty,
                        "cum_qty": cum_qty,
                        "side": order.get("Side"),
                    }
                )
            if log:
                logger.info(f"⏳ 未約定(予約・待機中)の注文を認識しました: {symbol}")

        return active_count, active_symbols, active_details, orders

    async def sync_positions(self, is_startup: bool = False, all_orders: Optional[List[dict]] = None) -> None:
        if is_startup:
            logger.info("🔄 持ち越しポジション（残高）を確認中...")

        product = 1 if self.config.TRADE_MODE == "CASH" else 2
        positions_data = await self.api.get_positions(product=product)
        if positions_data is None:
            return
        if not isinstance(positions_data, list):
            logger.warning("⚠️ positions API の返却形式が不正です")
            return

        current_hold_ids = {
            str(position.get("HoldID", position.get("ExecutionID", ""))): position
            for position in positions_data
        }

        for symbol, position in list(self.positions.items()):
            if not position.hold_id or position.hold_id in current_hold_ids:
                continue

            logger.info(f"💨 決済完了(ポジション消滅)を確認: {symbol}")
            pending = self._find_best_pending(symbol, "SELL")
            actual_price = None

            if pending and all_orders:
                for api_order in all_orders:
                    if str(api_order.get("ID")) == pending.order_id:
                        actual_price = self.market_data.extract_actual_price(api_order)
                        break

            if pending:
                slippage_yen = ""
                if actual_price is not None and pending.expected_bid > 0:
                    slippage_yen = pending.expected_bid - actual_price
                self.execution_logger.log_trade_execution(
                    {
                        "order_id": pending.order_id,
                        "execution_id": position.hold_id,
                        "order_sent_time": pending.order_sent_time,
                        "fill_time": now_str(),
                        "execution_order": pending.execution_order,
                        "symbol": symbol,
                        "side": "SELL",
                        "expected_ask": pending.expected_ask,
                        "expected_bid": pending.expected_bid,
                        "actual_price": actual_price if actual_price is not None else "",
                        "qty": position.qty,
                        "spread_pct": pending.spread_pct,
                        "slippage_yen": slippage_yen,
                    }
                )
                if actual_price is not None:
                    logger.info(
                        f"📝 SELLログ記録: {symbol} 予想Bid:{pending.expected_bid}円 "
                        f"-> 実約定:{actual_price}円 / Slip:{slippage_yen}円"
                    )
                else:
                    logger.info(f"📝 SELLログ記録: {symbol} 実約定単価は未取得")

            self.positions.pop(symbol, None)

        found_positions = False
        for position in positions_data:
            symbol = position.get("Symbol")
            qty = safe_int(position.get("LeavesQty"), 0) + safe_int(position.get("HoldQty"), 0)
            entry_price = safe_float(position.get("Price"), 0.0)
            hold_id = str(position.get("HoldID", position.get("ExecutionID", "")))
            exchange = as_int(position.get("Exchange"), 1)
            if exchange == 9:
                exchange = 1

            if qty <= 0 or not symbol:
                continue

            found_positions = True
            if symbol not in self.positions:
                self.add_position(symbol, qty, entry_price, exchange, hold_id)

                if hold_id and hold_id not in self.seen_exec_ids:
                    self.seen_exec_ids.add(hold_id)
                    pending = self._find_best_pending(symbol, "BUY")
                    if pending:
                        slippage_yen = ""
                        if entry_price > 0 and pending.expected_ask > 0:
                            slippage_yen = entry_price - pending.expected_ask
                        self.execution_logger.log_trade_execution(
                            {
                                "order_id": pending.order_id,
                                "execution_id": hold_id,
                                "order_sent_time": pending.order_sent_time,
                                "fill_time": now_str(),
                                "execution_order": pending.execution_order,
                                "symbol": symbol,
                                "side": "BUY",
                                "expected_ask": pending.expected_ask,
                                "expected_bid": pending.expected_bid,
                                "actual_price": entry_price if entry_price > 0 else "",
                                "qty": qty,
                                "spread_pct": pending.spread_pct,
                                "slippage_yen": slippage_yen,
                            }
                        )
                        logger.info(
                            f"📝 BUYログ記録: {symbol} 予想Ask:{pending.expected_ask}円 "
                            f"-> 実約定:{entry_price}円 / Slip:{slippage_yen}円"
                        )
                    else:
                        self.execution_logger.log_trade_execution(
                            {
                                "order_id": "",
                                "execution_id": hold_id,
                                "order_sent_time": "",
                                "fill_time": now_str(),
                                "execution_order": 0,
                                "symbol": symbol,
                                "side": "BUY",
                                "expected_ask": "",
                                "expected_bid": "",
                                "actual_price": entry_price if entry_price > 0 else "",
                                "qty": qty,
                                "spread_pct": "",
                                "slippage_yen": "",
                            }
                        )

                if not is_startup:
                    logger.info(
                        f"🎉 注文の約定を確認しました！監視モードに移行します: "
                        f"{symbol} (建玉ID: {hold_id})"
                    )
                else:
                    logger.info(f"📥 既存ポジションを認識しました: {symbol} (建玉ID: {hold_id})")
            else:
                existing = self.positions[symbol]
                if not existing.hold_id and hold_id:
                    existing.hold_id = hold_id
                    existing.entry_price = entry_price
                    existing.highest_price = entry_price
                    existing.stop_loss_price = entry_price * (1 - self.config.STOP_LOSS_PCT)
                    logger.info(f"🔄 約定完了！ {symbol} の情報を最新化しました(建玉ID: {hold_id})")

        if is_startup and not found_positions:
            logger.info("保有しているポジションはありませんでした。")

    async def cleanup_pendings(self, all_orders: Optional[List[dict]]) -> None:
        current_time = time.time()
        orders_by_id = {str(order.get("ID")): order for order in (all_orders or [])}

        for symbol in list(self.pending_orders.keys()):
            active_pendings = []
            for pending in self.pending_orders[symbol]:
                if current_time - pending.time_added <= self.PENDING_TTL_SECONDS:
                    active_pendings.append(pending)
                    continue

                status = "TIMEOUT"
                reason = "TTL (10min) expired"
                api_order = orders_by_id.get(pending.order_id)
                if api_order:
                    state = safe_int(api_order.get("State"), 0)
                    cum_qty = safe_int(api_order.get("CumQty"), 0)
                    if state == 5:
                        if cum_qty == 0:
                            status = "CANCELED/REJECTED"
                            reason = "Order completed with 0 qty"
                        else:
                            status = "PARTIAL_FILLED"
                            reason = f"Completed with {cum_qty} qty"
                    else:
                        status = f"API_STATE_{state}"

                self.execution_logger.log_order_status(
                    {
                        "order_id": pending.order_id,
                        "symbol": symbol,
                        "side": pending.side,
                        "expected_ask": pending.expected_ask,
                        "expected_bid": pending.expected_bid,
                        "order_sent_time": pending.order_sent_time,
                        "status": status,
                        "reason": reason,
                    }
                )
                logger.info(f"🗑️ ログ待機破棄: {symbol} ({status})")

            if active_pendings:
                self.pending_orders[symbol] = active_pendings
            else:
                self.pending_orders.pop(symbol, None)

    async def check_barriers(self) -> None:
        for symbol, position in list(self.positions.items()):
            if position.is_closing:
                continue

            board = await self.market_data.safe_get_board(symbol, position.exchange)
            if board is None:
                continue

            current_price = self.market_data.extract_reference_price(board)
            if current_price is None or current_price <= 0:
                continue

            if current_price > position.highest_price:
                position.highest_price = current_price
                if current_price > position.entry_price:
                    trail_pct = (
                        self.config.ATR_K2 * position.atr
                        if position.atr > 0
                        else self.config.STOP_LOSS_PCT
                    )
                    new_stop_loss = current_price * (1 - trail_pct)
                    if new_stop_loss > position.stop_loss_price:
                        position.stop_loss_price = new_stop_loss
                        logger.info(
                            f"✨ トレール発動！ {symbol} が最高値を更新。"
                            f"損切ラインを {new_stop_loss:,.1f}円 に引き上げました。"
                        )

            if current_price <= position.stop_loss_price:
                if current_price > position.entry_price:
                    logger.warning(f"📈 トレール利確！ {symbol} の決済（売り）を実行します。")
                else:
                    logger.error(f"📉 損切バリア到達！ {symbol} の決済（売り）を実行します。")
                await self.execute_exit(position)

    async def execute_exit(self, position: Position) -> None:
        position.is_closing = True
        board = await self.market_data.safe_get_board(position.symbol, self.config.EXCHANGE)
        ask1, bid1 = self.market_data.extract_best_bid_ask(board)
        exit_estimate = bid1 if bid1 > 0 else self.market_data.extract_reference_price(board)

        if exit_estimate is not None:
            realized = (exit_estimate - position.entry_price) * position.qty
            self.trading_state.today_realized_pnl += realized
            if self.trading_state.today_realized_pnl <= self.trading_state.daily_loss_limit:
                self.trading_state.trading_halted = True
                logger.error(
                    "🛑 KillSwitch発動: "
                    f"本日確定損益={self.trading_state.today_realized_pnl:,.0f}円"
                    f"（閾値={self.trading_state.daily_loss_limit:,.0f}円）"
                )
        else:
            logger.warning(
                f"⚠️ 決済前の基準価格が取得できなかったため、KillSwitch損益は更新しません: {position.symbol}"
            )

        result = await self.api.send_order(
            position.symbol,
            side="1",
            qty=position.qty,
            is_close=True,
            hold_id=position.hold_id,
            exchange=1,
        )
        if result and result.get("Result") == 0:
            order_id = str(result.get("OrderId", ""))
            logger.info(f"✅ 決済注文受付成功: {position.symbol} (OrderID: {order_id})")

            if not self.config.IS_PRODUCTION and order_id.startswith("SIM_"):
                self.execution_logger.log_order_status(
                    {
                        "order_id": order_id,
                        "symbol": position.symbol,
                        "side": "SELL",
                        "expected_ask": ask1,
                        "expected_bid": bid1,
                        "order_sent_time": now_str(),
                        "status": "SIM_SENT",
                        "reason": "Simulated close order (no execution)",
                    }
                )
                return

            self.register_pending(
                position.symbol,
                order_id=order_id,
                order_sent_time=now_str(),
                execution_order=0,
                expected_ask=ask1,
                expected_bid=bid1,
                side="SELL",
            )
            return

        position.is_closing = False
        logger.error(f"❌ 決済注文失敗: {result}")


def get_ticker_mapping() -> Dict[str, str]:
    mapping = {}
    if not os.path.exists("tickers.txt"):
        return mapping

    with open("tickers.txt", "r", encoding="utf-8") as handle:
        for line in handle:
            if "," not in line:
                continue
            name, ticker = line.split(",", 1)
            mapping[name.strip()] = ticker.replace(".T", "").strip()
    return mapping


def load_ai_signals(config: Config) -> List[dict]:
    signals = []
    if not os.path.exists(config.RECO_CSV_PATH):
        return signals

    mapping = get_ticker_mapping()
    with open(config.RECO_CSV_PATH, "r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        has_net_score = "Net_Score(%)" in fieldnames

        for row in reader:
            name = row.get("銘柄名", "")
            code = row.get("銘柄コード") or mapping.get(name)
            if not code:
                continue

            symbol = str(code).replace(".T", "").strip()
            atr = safe_float(row.get("ATR_Prev_Ratio"), 0.0)

            if has_net_score:
                net_pct = safe_float(row.get("Net_Score(%)"), 0.0)
                if net_pct <= 0:
                    continue
                prob = safe_float(row.get("メタ確信度(%)", row.get("メタ確信度", 50.0)), 50.0)
                prob = max(0.0, min(prob, 100.0))
                signals.append(
                    {
                        "name": name,
                        "symbol": symbol,
                        "net_pct": net_pct,
                        "prob": prob,
                        "atr": atr,
                    }
                )
                continue

            prob = safe_float(row.get("短期スコア"), 0.0)
            prob = max(0.0, min(prob, 100.0))
            if prob >= config.ENTRY_THRESHOLD_PROB:
                signals.append({"name": name, "symbol": symbol, "prob": prob, "atr": atr})

    if signals and "net_pct" in signals[0]:
        signals.sort(key=lambda signal: signal["net_pct"], reverse=True)
    else:
        signals.sort(key=lambda signal: signal.get("prob", 0.0), reverse=True)
    return signals


class TradingEngine:
    FORCE_EXIT_CONFIRM_TTL_SEC = 30
    FORCE_EXIT_RETRY_LIMIT = 3

    def __init__(
        self,
        config: Config,
        api: KabuAPI,
        market_data: MarketData,
        portfolio: PortfolioManager,
    ):
        self.config = config
        self.api = api
        self.market_data = market_data
        self.portfolio = portfolio

    async def run(self) -> None:
        logger.info(f"=== 自動取引エンジン起動 (モード: {self.config.TRADE_STYLE.upper()}) ===")
        await self.api.start_session()

        try:
            await self.api.get_token()
            if not self.api.token:
                return

            await self.portfolio.sync_positions(is_startup=True)
            active_orders_count, active_order_symbols, _, _ = await self.portfolio.fetch_active_orders(log=True)

            signals = self._load_entry_signals()
            if not signals:
                logger.info("😴 本日は新規エントリー条件を満たす銘柄はありませんでした。")
            else:
                active_orders_count = await self._process_entry_signals(
                    signals,
                    active_orders_count,
                    active_order_symbols,
                )

            if self.portfolio.positions or active_orders_count > 0:
                await self._monitor_positions()
        except KeyboardInterrupt:
            logger.info("システムを停止します...")
        finally:
            await self.api.close_session()

    def _load_entry_signals(self) -> List[dict]:
        now = datetime.now()
        if self.config.TRADE_STYLE == "day" and (now.hour > 14 or (now.hour == 14 and now.minute >= 30)):
            logger.info("🕒 14:30を過ぎているため、本日の新規エントリーは見送ります（デイトレモード）。")
            self.portfolio.set_active_signals([])
            return []

        signals = load_ai_signals(self.config)
        self.portfolio.set_active_signals(signals)
        return signals

    async def _get_available_cash(self) -> float:
        wallet = await self.api.get_wallet_cash()
        if not isinstance(wallet, dict):
            return 1_000_000.0
        return safe_float(wallet.get("StockAccountWallet"), 1_000_000.0)

    async def _calculate_order_qty(self, signal: dict, reference_price: float) -> int:
        if reference_price <= 0:
            return 0

        available_cash = await self._get_available_cash()
        position_limit = max(self.config.MAX_POSITIONS, 1)
        qty = 0

        if self.config.LOT_CALC_MODE == "FIXED":
            qty = self.config.FIXED_LOT_SIZE
        elif self.config.LOT_CALC_MODE == "AUTO":
            qty = int((available_cash * self.config.AUTO_INVEST_RATIO) / position_limit / reference_price)
            qty = (qty // 100) * 100
        elif self.config.LOT_CALC_MODE == "KELLY":
            b = self.config.TAKE_PROFIT_PCT / self.config.STOP_LOSS_PCT if self.config.STOP_LOSS_PCT > 0 else 1.0
            prob_pct = max(0.0, min(safe_float(signal.get("prob"), 50.0), 100.0))
            p = prob_pct / 100.0
            kelly_fraction = p - ((1.0 - p) / b) if b > 0 else 0.0
            invest_ratio = max(0.0, min(kelly_fraction / 2.0, self.config.AUTO_INVEST_RATIO))
            if invest_ratio > 0:
                qty = int((available_cash * invest_ratio) / reference_price)
                qty = (qty // 100) * 100
                logger.info(
                    f"🧠 ケリー基準計算: 勝率={p * 100:.1f}%, "
                    f"投資割合={invest_ratio * 100:.1f}% -> 算出ロット={qty}株"
                )
            else:
                logger.warning(f"⚠️ ケリー基準がマイナスのため見送ります。(勝率: {p * 100:.1f}%)")

        max_notional = available_cash * 0.10
        max_qty_notional = int(max_notional // reference_price) if reference_price > 0 else 0

        stop_pct = self.config.STOP_LOSS_PCT if self.config.STOP_LOSS_PCT > 0 else 0.05
        max_risk = available_cash * 0.005
        max_qty_risk = int(max_risk // (reference_price * stop_pct)) if reference_price > 0 else 0

        max_qty = max(0, min(max_qty_notional, max_qty_risk))
        max_qty = (max_qty // 100) * 100
        return min(qty, max_qty)

    async def _process_entry_signals(
        self,
        signals: List[dict],
        active_orders_count: int,
        active_order_symbols: List[str],
    ) -> int:
        execution_order = 0
        consecutive_board_failures = 0

        for signal in signals:
            if self.portfolio.trading_state.trading_halted:
                halt_reason = self.portfolio.trading_state.halt_reason or "TRADING_HALTED"
                logger.warning(f"🛑 Halt中のため新規エントリーを停止します: {halt_reason}")
                break
            if signal["symbol"] in self.portfolio.positions:
                continue
            if signal["symbol"] in active_order_symbols:
                continue
            if len(self.portfolio.positions) + active_orders_count >= self.config.MAX_POSITIONS:
                logger.warning(
                    f"🚫 最大保有数({self.config.MAX_POSITIONS}銘柄)に達したため、"
                    "これ以上の新規エントリーを見送ります。"
                )
                break

            logger.info(f"🌟 AIシグナル発火！ 銘柄: {signal['name']} ({signal['symbol']})")
            execution_order += 1

            board = await self.market_data.safe_get_board(signal["symbol"], self.config.EXCHANGE)
            if board is None:
                consecutive_board_failures += 1
                logger.warning(f"⚠️ 板情報が取得できませんでした: {signal['symbol']}（スキップ）")
                if consecutive_board_failures >= 5:
                    logger.error("🛑 板情報取得の連続失敗が閾値に達したため、本日の新規エントリーを停止します")
                    break
                continue

            consecutive_board_failures = 0
            ask1, bid1 = self.market_data.extract_best_bid_ask(board)
            if ask1 <= 0:
                logger.warning(f"⚠️ 最良売気配(Ask)が0です: {signal['symbol']}（スキップ）")
                continue

            qty = await self._calculate_order_qty(signal, ask1)
            if qty <= 0:
                logger.info("⛔ 1銘柄上限によりロットが0になったためスキップ")
                continue

            order_sent_time = now_str()
            result = await self.api.send_order(signal["symbol"], side="2", qty=qty, is_close=False)
            if result and result.get("Result") == 0:
                order_id = str(result.get("OrderId", ""))
                logger.info(f"✅ エントリー注文送信成功 order_id={order_id}")

                if not self.config.IS_PRODUCTION and order_id.startswith("SIM_"):
                    self.portfolio.execution_logger.log_order_status(
                        {
                            "order_id": order_id,
                            "symbol": signal["symbol"],
                            "side": "BUY",
                            "expected_ask": ask1,
                            "expected_bid": bid1,
                            "order_sent_time": order_sent_time,
                            "status": "SIM_SENT",
                            "reason": "Simulated order (no execution)",
                        }
                    )
                else:
                    self.portfolio.register_pending(
                        signal["symbol"],
                        order_id=order_id,
                        order_sent_time=order_sent_time,
                        execution_order=execution_order,
                        expected_ask=ask1,
                        expected_bid=bid1,
                        side="BUY",
                    )
                    active_orders_count += 1
            else:
                logger.warning(f"⚠️ 注文送信に失敗しました。レスポンス: {result}")

        return active_orders_count

    def _get_positions_product(self) -> int:
        return 1 if self.config.TRADE_MODE == "CASH" else 2

    async def _request_force_exit_for_position(self, position: Any, local_state: dict, reason: str) -> dict:
        board = await self.market_data.safe_get_board(
            local_state.get("symbol", getattr(position, "symbol", "")),
            local_state.get("exchange", getattr(position, "exchange", self.config.EXCHANGE)),
        )
        ask1, bid1 = self.market_data.extract_best_bid_ask(board)

        result = await request_exit(self.api, position, local_state, reason)
        order_id = result.get("exit_order_id", "")
        symbol = local_state.get("symbol", getattr(position, "symbol", ""))

        if result.get("ok"):
            logger.info(
                f"🚪 強制決済注文を送信しました: {symbol} "
                f"attempt={result['attempt_no']} order_id={order_id} reason={reason}"
            )
            if not self.config.IS_PRODUCTION and order_id.startswith("SIM_"):
                self.portfolio.execution_logger.log_order_status(
                    {
                        "order_id": order_id,
                        "symbol": symbol,
                        "side": "SELL",
                        "expected_ask": ask1,
                        "expected_bid": bid1,
                        "order_sent_time": result["requested_at"],
                        "status": "SIM_SENT",
                        "reason": reason,
                    }
                )
            else:
                self.portfolio.register_pending(
                    symbol,
                    order_id=order_id,
                    order_sent_time=result["requested_at"],
                    execution_order=0,
                    expected_ask=ask1,
                    expected_bid=bid1,
                    side="SELL",
                )
            return result

        logger.error(
            f"❌ 強制決済注文の送信に失敗: {symbol} "
            f"attempt={result['attempt_no']} reason={reason} error={result['error']}"
        )
        return result

    async def _start_force_exit_requests(self, force_exit_states: Dict[str, dict]) -> None:
        logger.warning("⏰ 14:50 を回りました。建玉ごとに返済注文を送信し、建玉消滅を確認します。")
        for symbol, position in list(self.portfolio.positions.items()):
            if position.is_closing:
                continue

            position.is_closing = True
            state_key = position.hold_id or symbol
            local_state = force_exit_states.setdefault(
                state_key,
                {
                    "symbol": position.symbol,
                    "hold_id": position.hold_id,
                    "position_side": "BUY",
                    "known_position_qty": position.qty,
                    "exchange": position.exchange,
                    "status": "FORCE_EXIT_PREPARED",
                    "exit_retry_limit": self.FORCE_EXIT_RETRY_LIMIT,
                    "halt_reason": "",
                    "unresolved_exit_count": 0,
                },
            )
            await self._request_force_exit_for_position(position, local_state, "14:50_force_exit")

    def _halt_on_unresolved_force_exit(self, local_state: dict, confirmation: dict) -> None:
        if local_state.get("_unresolved_logged"):
            return

        trading_state = self.portfolio.trading_state
        trading_state.trading_halted = True
        trading_state.halt_reason = HALT_FORCE_EXIT_UNRESOLVED
        trading_state.unresolved_exit_count += 1

        local_state["status"] = "EXIT_UNRESOLVED"
        local_state["halt_reason"] = HALT_FORCE_EXIT_UNRESOLVED
        local_state["unresolved_exit_count"] = safe_int(
            _pick_first_value(local_state, ["unresolved_exit_count"]),
            0,
        ) + 1

        reason_detail = confirmation.get("reason", "")
        logger.error(
            f"🛑 強制決済が未解消のため新規エントリーを停止します: "
            f"symbol={local_state.get('symbol', '')} reason={HALT_FORCE_EXIT_UNRESOLVED} detail={reason_detail}"
        )
        self.portfolio.execution_logger.log_order_status(
            {
                "order_id": _safe_str(_pick_first_value(local_state, ["exit_order_id"])),
                "symbol": local_state.get("symbol", ""),
                "side": "SELL",
                "expected_ask": "",
                "expected_bid": "",
                "order_sent_time": _safe_str(
                    _pick_first_value(local_state, ["exit_requested_at", "last_exit_attempt_at"])
                ),
                "status": "HALT",
                "reason": f"{HALT_FORCE_EXIT_UNRESOLVED}: {reason_detail}",
            }
        )
        local_state["_unresolved_logged"] = True

    async def _confirm_force_exit_states(self, force_exit_states: Dict[str, dict]) -> None:
        if not force_exit_states:
            return

        orders_snapshot = await fetch_orders_snapshot(self.api)
        positions_snapshot = await fetch_positions_snapshot(self.api, product=self._get_positions_product())

        for state_key, local_state in list(force_exit_states.items()):
            confirmation = confirm_exit(
                local_state,
                orders_snapshot=orders_snapshot,
                positions_snapshot=positions_snapshot,
                ttl_sec=self.FORCE_EXIT_CONFIRM_TTL_SEC,
            )
            result = confirmation["result"]
            remaining_qty = safe_int(confirmation.get("remaining_qty"), 0)
            last_result = local_state.get("_last_confirm_result")
            last_remaining = safe_int(local_state.get("_last_confirm_remaining_qty"), -1)

            if result in {"PARTIAL", "PENDING", "RETRY", "UNRESOLVED"}:
                local_state["known_position_qty"] = remaining_qty or safe_int(
                    _pick_first_value(local_state, ["known_position_qty"]),
                    0,
                )

            if result != last_result or remaining_qty != last_remaining:
                logger.info(
                    f"🔎 強制決済確認: symbol={local_state.get('symbol', '')} "
                    f"result={result} remaining_qty={remaining_qty} reason={confirmation['reason']}"
                )
                local_state["_last_confirm_result"] = result
                local_state["_last_confirm_remaining_qty"] = remaining_qty

            if result == "CLOSED":
                local_state["status"] = "CLOSED"
                logger.info(
                    f"✅ 強制決済完了を確認: {local_state.get('symbol', '')} "
                    f"hold_id={local_state.get('hold_id', '')}"
                )
                force_exit_states.pop(state_key, None)
                continue

            if result == "PARTIAL":
                local_state["status"] = "EXIT_PARTIAL"
                continue

            if result == "PENDING":
                local_state["status"] = "EXIT_PENDING"
                continue

            if result == "RETRY":
                local_state["status"] = "EXIT_RETRYING"
                retry_position = {
                    "symbol": local_state.get("symbol", ""),
                    "hold_id": local_state.get("hold_id", ""),
                    "qty": remaining_qty or safe_int(_pick_first_value(local_state, ["known_position_qty"]), 0),
                    "exchange": as_int(_pick_first_value(local_state, ["exchange"]), self.config.EXCHANGE),
                    "side": "BUY",
                }
                await self._request_force_exit_for_position(retry_position, local_state, "14:50_force_exit_retry")
                continue

            if result == "UNRESOLVED":
                self._halt_on_unresolved_force_exit(local_state, confirmation)

    async def _monitor_positions(self) -> None:
        has_force_closed_today = False
        logger.info("=== ⏱ リアルタイムインメモリ監視ループ開始 ===")
        sync_counter = 0
        last_logged_active_prices: Dict[str, float] = {}
        force_exit_states: Dict[str, dict] = {}

        while True:
            now = datetime.now()
            if self.config.TRADE_STYLE == "day" and now.hour == 14 and now.minute >= 50:
                if not has_force_closed_today:
                    await self._start_force_exit_requests(force_exit_states)
                    has_force_closed_today = True
                await self._confirm_force_exit_states(force_exit_states)
                if not self.portfolio.positions:
                    break

            if now.hour >= 15:
                logger.info("🌙 15:00を回りました。本日の市場監視を終了します。")
                break

            active_count, _, active_details, all_orders = await self.portfolio.fetch_active_orders(log=False)
            sync_counter += 1
            if sync_counter >= 3:
                await self.portfolio.sync_positions(is_startup=False, all_orders=all_orders)
                sync_counter = 0

            await self.portfolio.cleanup_pendings(all_orders)

            for detail in active_details:
                symbol = detail["symbol"]
                _, current_price = await self.market_data.safe_get_price(symbol, self.config.EXCHANGE)
                if current_price is not None and current_price != last_logged_active_prices.get(symbol):
                    last_logged_active_prices[symbol] = current_price

            if not has_force_closed_today:
                await self.portfolio.check_barriers()

            if not self.portfolio.positions and self.config.TRADE_STYLE == "day" and active_count == 0:
                await self.portfolio.sync_positions(is_startup=False, all_orders=all_orders)
                if not self.portfolio.positions:
                    logger.info("📉 全てのポジションの決済が完了し、未約定の注文もありません。本日の取引を終了します。")
                    break

            await asyncio.sleep(5)


async def main() -> None:
    config = Config()
    _, _, _, _, engine = build_components(config)
    await engine.run()


def build_components(config: Config):
    trading_state = TradingState()
    api = KabuAPI(config)
    market_data = MarketData(api)
    execution_logger = ExecutionLogger(is_sim=not config.IS_PRODUCTION)
    portfolio = PortfolioManager(
        config=config,
        api=api,
        market_data=market_data,
        execution_logger=execution_logger,
        trading_state=trading_state,
    )
    engine = TradingEngine(config, api, market_data, portfolio)
    return trading_state, api, market_data, portfolio, engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="設定読込と主要コンポーネント初期化のみ実施し、API通信は行いません。",
    )
    return parser.parse_args()


def is_placeholder_secret(value: str) -> bool:
    stripped = value.strip()
    if stripped.lower() in PLACEHOLDER_VALUES:
        return True
    return any(stripped.upper().startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)


def validate_dry_run_config(config: Config) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not os.path.exists(config.config_file):
        errors.append(f"{config.config_file} が見つかりません。")
    if not os.path.exists(config.local_config_file):
        errors.append(f"{config.local_config_file} が見つかりません。")

    selected_password_key = "API_PASSWORD_PROD" if config.IS_PRODUCTION else "API_PASSWORD_SIM"
    selected_api_password = str(config.config_data.get(selected_password_key, config.API_PASSWORD) or "")
    trade_password = str(config.TRADE_PASSWORD or "")

    if not selected_api_password.strip():
        errors.append(f"{selected_password_key} が空です。")
    elif is_placeholder_secret(selected_api_password):
        warnings.append(f"{selected_password_key} がプレースホルダー値の可能性があります。")

    if not trade_password.strip():
        errors.append("TRADE_PASSWORD が空です。")
    elif is_placeholder_secret(trade_password):
        warnings.append("TRADE_PASSWORD がプレースホルダー値の可能性があります。")

    return errors, warnings


def run_dry_run() -> int:
    logger.info("=== auto_trade.py dry-run 開始 ===")
    logger.info("dry-run では API セッション開始・トークン取得・板取得・注文・positions 取得を行いません。")

    try:
        config = Config()
        errors, warnings = validate_dry_run_config(config)

        mode = "PROD" if config.IS_PRODUCTION else "SIM"
        selected_password_key = "API_PASSWORD_PROD" if config.IS_PRODUCTION else "API_PASSWORD_SIM"
        trading_state, api, market_data, portfolio, engine = build_components(config)
        execution_logger = portfolio.execution_logger
        execution_logger.ensure_log_headers()

        logger.info(f"設定読込: settings={config.config_file}, local={config.local_config_file}")
        logger.info(f"選択環境: {mode} (IS_PRODUCTION={config.IS_PRODUCTION})")
        logger.info(f"API URL: {config.API_URL}")
        logger.info(f"選択APIパスワードキー: {selected_password_key}")
        logger.info(f"アプリログ: {log_filename}")
        logger.info(f"約定ログ: {execution_logger.exec_log_path}")
        logger.info(f"注文状態ログ: {execution_logger.status_log_path}")
        logger.info(f"recommendations.csv: {config.RECO_CSV_PATH}")

        health_log_path = config.config_data.get("HEALTH_LOG_PATH")
        if health_log_path:
            logger.info(f"daily_health_log.csv: {health_log_path}")

        if "BREAKER_ENABLED" in config.config_data:
            logger.info(
                "breaker: "
                f"enabled={config.config_data.get('BREAKER_ENABLED')} "
                f"ticker={config.config_data.get('BREAKER_TICKER')} "
                f"ret_threshold={config.config_data.get('BREAKER_RET_THRESHOLD')} "
                f"ma_days={config.config_data.get('BREAKER_MA_DAYS')}"
            )

        logger.info(
            "初期化済みコンポーネント: "
            f"{api.__class__.__name__}, {market_data.__class__.__name__}, "
            f"{execution_logger.__class__.__name__}, {portfolio.__class__.__name__}, "
            f"{engine.__class__.__name__}, {trading_state.__class__.__name__}"
        )

        for warning in warnings:
            logger.warning(f"dry-run 警告: {warning}")

        if errors:
            for error in errors:
                logger.error(f"dry-run エラー: {error}")
            logger.error("dry-run は設定不備のため失敗しました。")
            return 1

        logger.info("dry-run 成功: API 通信なしで設定読込と主要コンポーネント初期化を確認しました。")
        return 0
    except Exception as exc:
        logger.exception(f"dry-run 失敗: {exc}")
        return 1


if __name__ == "__main__":
    args = parse_args()

    if args.dry_run:
        sys.exit(run_dry_run())

    if sys.platform == "win32" and sys.version_info < (3, 14):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except AttributeError:
            pass

    asyncio.run(main())
