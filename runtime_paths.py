from datetime import datetime
from pathlib import Path


RUNTIME_DIR = Path("runtime")
RUNTIME_HEALTH_DIR = RUNTIME_DIR / "health"
RUNTIME_ORDERS_DIR = RUNTIME_DIR / "orders"
RUNTIME_SIGNALS_DIR = RUNTIME_DIR / "signals"
RUNTIME_LOGS_DIR = RUNTIME_DIR / "logs"

DAILY_HEALTH_LOG_PATH = str(RUNTIME_HEALTH_DIR / "daily_health_log.csv")
RECOMMENDATIONS_PATH = str(RUNTIME_SIGNALS_DIR / "recommendations.csv")
PREDICTION_HISTORY_PATH = str(RUNTIME_SIGNALS_DIR / "prediction_history.csv")
BREAKER_EVENT_LOG_PATH = str(RUNTIME_SIGNALS_DIR / "breaker_event_log.csv")
SIMULATED_ORDER_LOG_PATH = str(RUNTIME_SIGNALS_DIR / "simulated_order_log.csv")

TRADE_EXECUTION_LOG_PATH = str(RUNTIME_ORDERS_DIR / "trade_execution_log.csv")
TRADE_EXECUTION_LOG_SIM_PATH = str(RUNTIME_ORDERS_DIR / "trade_execution_log_SIM.csv")
ORDER_STATUS_LOG_PATH = str(RUNTIME_ORDERS_DIR / "order_status_log.csv")
ORDER_STATUS_LOG_SIM_PATH = str(RUNTIME_ORDERS_DIR / "order_status_log_SIM.csv")

REPORTS_OUTPUT_DIR = str(RUNTIME_LOGS_DIR)


def ensure_runtime_dirs() -> None:
    for directory in (
        RUNTIME_DIR,
        RUNTIME_HEALTH_DIR,
        RUNTIME_ORDERS_DIR,
        RUNTIME_SIGNALS_DIR,
        RUNTIME_LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def build_auto_trade_log_path(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return str(RUNTIME_LOGS_DIR / f"auto_trade_{now.strftime('%Y%m%d')}.log")
