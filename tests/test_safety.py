import os
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta

from auto_trade import (
    HALT_API_BOARD_FAILURE,
    HALT_API_ORDER_ID_MISSING,
    HALT_API_ORDERS_FAILURE,
    HALT_API_POSITIONS_FAILURE,
    HALT_FORCE_EXIT_UNRESOLVED,
    ExecutionLogger,
    MarketData,
    PortfolioManager,
    Position,
    TradingEngine,
    TradingState,
    confirm_exit,
    reconcile_trade_state,
    request_exit,
)


def now_minus(seconds: int) -> str:
    return (datetime.now() - timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def now_plus(seconds: int) -> str:
    return (datetime.now() + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def make_order_snapshot(
    order_id: str,
    symbol: str = "7203",
    side: str = "SELL",
    state: str = "ORDERED",
    qty: int = 100,
    cum_qty: int = 0,
    remaining_qty: int = 100,
):
    return {
        "order_id": order_id,
        "symbol": symbol,
        "side": side,
        "state": state,
        "qty": qty,
        "cum_qty": cum_qty,
        "remaining_qty": remaining_qty,
        "price": 0.0,
        "avg_price": 0.0,
        "sent_at": now_minus(1),
        "raw": {},
    }


def make_position_snapshot(
    hold_id: str,
    symbol: str = "7203",
    side: str = "BUY",
    qty: int = 100,
    leaves_qty: int = None,
):
    if leaves_qty is None:
        leaves_qty = qty
    return {
        "hold_id": hold_id,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": 2500.0,
        "leaves_qty": leaves_qty,
        "state": "OPEN" if leaves_qty > 0 else "CLOSED",
        "raw": {},
    }


def make_exit_state(
    hold_id: str = "H001",
    symbol: str = "7203",
    known_position_qty: int = 100,
    exit_order_id: str = "OID-1",
    attempt_no: int = 1,
    requested_at: str = None,
    retry_limit: int = 3,
):
    return {
        "symbol": symbol,
        "hold_id": hold_id,
        "position_side": "BUY",
        "known_position_qty": known_position_qty,
        "qty": known_position_qty,
        "exit_order_id": exit_order_id,
        "exit_attempt_no": attempt_no,
        "exit_retry_limit": retry_limit,
        "exit_requested_at": requested_at or now_minus(1),
        "status": "EXIT_SENT",
    }


@dataclass
class DummyConfig:
    IS_PRODUCTION: bool = False
    EXCHANGE: int = 1
    TRADE_MODE: str = "MARGIN"
    STOP_LOSS_PCT: float = 0.05
    ATR_K1: float = 2.0
    ATR_K2: float = 3.0
    MAX_POSITIONS: int = 2
    LOT_CALC_MODE: str = "FIXED"
    FIXED_LOT_SIZE: int = 100
    AUTO_INVEST_RATIO: float = 0.3
    TAKE_PROFIT_PCT: float = 0.05
    ENTRY_THRESHOLD_PROB: float = 55.0
    TRADE_STYLE: str = "day"
    ACCOUNT_TYPE: int = 4
    TRADE_PASSWORD: str = "dummy"
    API_URL: str = "http://localhost:18081/kabusapi"
    RECO_CSV_PATH: str = "recommendations.csv"


class SafetyTestMixin:
    def setUp(self):
        super().setUp()
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)

    def make_logger(self):
        logger = ExecutionLogger(is_sim=True)
        logger.exec_log_path = os.path.join(self._temp_dir.name, "trade_execution.csv")
        logger.status_log_path = os.path.join(self._temp_dir.name, "order_status.csv")
        return logger

    def make_market(self, api, trading_state, execution_logger):
        return MarketData(api, trading_state=trading_state, execution_logger=execution_logger)

    def make_portfolio(self, api, trading_state, execution_logger, config=None):
        if config is None:
            config = DummyConfig()
        market = self.make_market(api, trading_state, execution_logger)
        portfolio = PortfolioManager(
            config=config,
            api=api,
            market_data=market,
            execution_logger=execution_logger,
            trading_state=trading_state,
        )
        return market, portfolio


class BoardFailureAPI:
    async def get_board(self, symbol, exchange):
        return None


class OrdersFailureAPI:
    async def get_orders(self, product=0):
        return None


class PositionsFailureAPI:
    async def get_positions(self, product=0):
        return None


class MissingOrderIdAPI:
    async def get_board(self, symbol, exchange):
        return {"Sell1": {"Price": 10}, "Buy1": {"Price": 9}, "CurrentPrice": 9.5}

    async def get_wallet_cash(self):
        return {"StockAccountWallet": 1_000_000}

    async def send_order(self, symbol, side, qty, price=0, is_close=False, hold_id=None, exchange=None):
        return {"Result": 0}


class ForceExitRetryAPI:
    def __init__(self):
        self.send_calls = []
        self.orders = []
        self.positions = []

    async def get_board(self, symbol, exchange):
        return {"Sell1": {"Price": 2501}, "Buy1": {"Price": 2499}, "CurrentPrice": 2500}

    async def send_order(self, symbol, side, qty, price=0, is_close=False, hold_id=None, exchange=None):
        order_id = f"OID-{len(self.send_calls) + 1}"
        self.send_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "hold_id": hold_id,
                "exchange": exchange,
                "is_close": is_close,
                "order_id": order_id,
            }
        )
        self.orders = [
            {
                "ID": order_id,
                "Symbol": symbol,
                "Side": side,
                "State": 2,
                "OrderQty": qty,
                "CumQty": 0,
                "LeavesQty": qty,
            }
        ]
        return {"Result": 0, "OrderId": order_id}

    async def get_orders(self, product=0):
        return list(self.orders)

    async def get_positions(self, product=0):
        return list(self.positions)


class RejectingExitAPI:
    async def send_order(self, symbol, side, qty, price=0, is_close=False, hold_id=None, exchange=None):
        return {"Result": 1, "Message": "rejected"}


class APISafetyTests(SafetyTestMixin, unittest.IsolatedAsyncioTestCase):
    async def test_board_api_failure_halts_after_three_failures(self):
        trading_state = TradingState()
        execution_logger = self.make_logger()
        market = self.make_market(BoardFailureAPI(), trading_state, execution_logger)

        for _ in range(3):
            board = await market.safe_get_board("7203", 1)
            self.assertIsNone(board)

        self.assertEqual(trading_state.board_failure_count, 3)
        self.assertTrue(trading_state.trading_halted)
        self.assertEqual(trading_state.halt_reason, HALT_API_BOARD_FAILURE)

    async def test_orders_api_failure_halts_after_three_failures(self):
        trading_state = TradingState()
        execution_logger = self.make_logger()
        _, portfolio = self.make_portfolio(OrdersFailureAPI(), trading_state, execution_logger)

        for _ in range(3):
            active_count, active_symbols, active_details, orders = await portfolio.fetch_active_orders(log=False)
            self.assertEqual((active_count, active_symbols, active_details, orders), (0, [], [], []))

        self.assertEqual(trading_state.orders_failure_count, 3)
        self.assertTrue(trading_state.trading_halted)
        self.assertEqual(trading_state.halt_reason, HALT_API_ORDERS_FAILURE)

    async def test_positions_api_failure_halts_after_three_failures(self):
        trading_state = TradingState()
        execution_logger = self.make_logger()
        _, portfolio = self.make_portfolio(PositionsFailureAPI(), trading_state, execution_logger)

        for _ in range(3):
            await portfolio.sync_positions(is_startup=False)

        self.assertEqual(trading_state.positions_failure_count, 3)
        self.assertTrue(trading_state.trading_halted)
        self.assertEqual(trading_state.halt_reason, HALT_API_POSITIONS_FAILURE)

    async def test_send_order_without_order_id_halts_trading(self):
        config = DummyConfig()
        trading_state = TradingState()
        execution_logger = self.make_logger()
        api = MissingOrderIdAPI()
        market = self.make_market(api, trading_state, execution_logger)
        portfolio = PortfolioManager(
            config=config,
            api=api,
            market_data=market,
            execution_logger=execution_logger,
            trading_state=trading_state,
        )
        engine = TradingEngine(config, api, market, portfolio)

        active_orders_count = await engine._process_entry_signals(
            [{"name": "Test", "symbol": "7203", "prob": 80.0, "atr": 0.0}],
            active_orders_count=0,
            active_order_symbols=[],
        )

        self.assertEqual(active_orders_count, 0)
        self.assertTrue(trading_state.trading_halted)
        self.assertEqual(trading_state.halt_reason, HALT_API_ORDER_ID_MISSING)

    async def test_force_exit_retry_then_unresolved_halts(self):
        config = DummyConfig()
        trading_state = TradingState()
        execution_logger = self.make_logger()
        api = ForceExitRetryAPI()
        market = self.make_market(api, trading_state, execution_logger)
        portfolio = PortfolioManager(
            config=config,
            api=api,
            market_data=market,
            execution_logger=execution_logger,
            trading_state=trading_state,
        )
        engine = TradingEngine(config, api, market, portfolio)
        engine.FORCE_EXIT_RETRY_LIMIT = 2
        engine.FORCE_EXIT_CONFIRM_TTL_SEC = 0

        position = Position(
            symbol="7203",
            qty=100,
            entry_price=2500.0,
            stop_loss_price=2400.0,
            highest_price=2550.0,
            hold_id="H001",
            exchange=1,
        )
        portfolio.positions[position.symbol] = position
        api.positions = [
            {
                "HoldID": "H001",
                "Symbol": "7203",
                "Side": "2",
                "HoldQty": 100,
                "LeavesQty": 100,
                "Price": 2500,
                "Exchange": 1,
            }
        ]

        force_exit_states = {}
        await engine._start_force_exit_requests(force_exit_states)
        local_state = force_exit_states["H001"]
        local_state["exit_requested_at"] = now_minus(60)

        await engine._confirm_force_exit_states(force_exit_states)
        self.assertEqual(len(api.send_calls), 2)
        self.assertEqual(force_exit_states["H001"]["exit_attempt_no"], 2)
        self.assertFalse(trading_state.trading_halted)

        force_exit_states["H001"]["exit_requested_at"] = now_minus(60)
        await engine._confirm_force_exit_states(force_exit_states)

        self.assertTrue(trading_state.trading_halted)
        self.assertEqual(trading_state.halt_reason, HALT_FORCE_EXIT_UNRESOLVED)
        self.assertEqual(trading_state.unresolved_exit_count, 1)
        self.assertEqual(force_exit_states["H001"]["status"], "EXIT_UNRESOLVED")

    async def test_request_exit_failure_still_increments_attempt_no(self):
        local_state = {"symbol": "7203", "hold_id": "H001", "position_side": "BUY", "exit_attempt_no": 1}
        result = await request_exit(
            RejectingExitAPI(),
            {"symbol": "7203", "hold_id": "H001", "qty": 100, "exchange": 1, "side": "BUY"},
            local_state,
            "retry",
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["attempt_no"], 2)
        self.assertEqual(local_state["exit_attempt_no"], 2)
        self.assertEqual(local_state["last_exit_reason"], "retry")
        self.assertEqual(local_state["last_exit_error"], "rejected")


class ExitStateTests(unittest.TestCase):
    def test_confirm_exit_returns_all_supported_results(self):
        scenarios = [
            (
                "CLOSED",
                make_exit_state(requested_at=now_minus(1)),
                {"OID-1": make_order_snapshot("OID-1")},
                {},
                30,
            ),
            (
                "PARTIAL",
                make_exit_state(known_position_qty=100, requested_at=now_minus(1)),
                {"OID-1": make_order_snapshot("OID-1", state="PARTIAL", cum_qty=60, remaining_qty=40)},
                {"H001": make_position_snapshot("H001", qty=40, leaves_qty=40)},
                30,
            ),
            (
                "PENDING",
                make_exit_state(requested_at=now_plus(30)),
                {"OID-1": make_order_snapshot("OID-1")},
                {"H001": make_position_snapshot("H001", qty=100, leaves_qty=100)},
                30,
            ),
            (
                "RETRY",
                make_exit_state(requested_at=now_minus(60), attempt_no=1, retry_limit=3),
                {"OID-1": make_order_snapshot("OID-1")},
                {"H001": make_position_snapshot("H001", qty=100, leaves_qty=100)},
                30,
            ),
            (
                "UNRESOLVED",
                make_exit_state(requested_at=now_minus(60), attempt_no=3, retry_limit=3),
                {"OID-1": make_order_snapshot("OID-1")},
                {"H001": make_position_snapshot("H001", qty=100, leaves_qty=100)},
                30,
            ),
        ]

        for expected, local_state, orders_snapshot, positions_snapshot, ttl_sec in scenarios:
            with self.subTest(expected=expected):
                result = confirm_exit(local_state, orders_snapshot, positions_snapshot, ttl_sec=ttl_sec)
                self.assertEqual(result["result"], expected)

    def test_reconcile_trade_state_returns_expected_codes(self):
        scenarios = [
            (
                "MATCHED_CLOSED",
                make_exit_state(requested_at=now_minus(1)),
                {"OID-1": make_order_snapshot("OID-1")},
                {},
            ),
            (
                "EXIT_PENDING",
                make_exit_state(requested_at=now_minus(1)),
                {"OID-1": make_order_snapshot("OID-1")},
                {"H001": make_position_snapshot("H001")},
            ),
            (
                "EXIT_PARTIAL",
                make_exit_state(requested_at=now_minus(1)),
                {"OID-1": make_order_snapshot("OID-1", state="PARTIAL", cum_qty=60, remaining_qty=40)},
                {"H001": make_position_snapshot("H001", qty=40, leaves_qty=40)},
            ),
            (
                "ENTRY_PENDING",
                {
                    "symbol": "7203",
                    "qty": 100,
                    "entry_order_id": "B-1",
                    "order_side": "BUY",
                    "phase": "ENTRY",
                    "status": "ENTRY_PENDING",
                },
                {"B-1": make_order_snapshot("B-1", side="BUY")},
                {},
            ),
            (
                "POSITION_ONLY",
                {"symbol": "7203", "hold_id": "H001", "qty": 100},
                {},
                {"H001": make_position_snapshot("H001")},
            ),
            (
                "ORDER_ONLY",
                {"symbol": "7203", "order_id": "O-1", "qty": 100, "status": "UNKNOWN"},
                {"O-1": make_order_snapshot("O-1", side="BUY")},
                {},
            ),
        ]

        for expected, local_state, orders_snapshot, positions_snapshot in scenarios:
            with self.subTest(expected=expected):
                result = reconcile_trade_state(local_state, orders_snapshot, positions_snapshot)
                self.assertEqual(result["result"], expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
