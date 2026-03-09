import os
import tempfile
import unittest

import pandas as pd

from daily_batch import (
    BREAKER_EVENT_LOG_COLUMNS,
    MarketBreakerResult,
    SIMULATED_ORDER_LOG_COLUMNS,
    append_observation_rows,
    build_breaker_observation_rows,
)


class BreakerObservationLogTests(unittest.TestCase):
    def test_build_breaker_observation_rows_for_blocked_candidates(self):
        breaker_result = MarketBreakerResult(
            breaker_enabled=True,
            breaker=True,
            reason="close_below_ma",
            ticker="1306.T",
            ma_days=5,
            ret_threshold=-0.015,
            close=2789.12,
            ma=2801.34,
            ret1=-0.01,
        )
        blocked_candidates = pd.DataFrame(
            [
                {
                    "Ticker": "Test Corp",
                    "TickerCode": "7203",
                    "Close": 2500.0,
                    "ATR_Prev_Ratio": 0.023,
                    "Net_Score": 0.0125,
                }
            ]
        )

        event_rows, simulated_rows = build_breaker_observation_rows(breaker_result, blocked_candidates)

        self.assertEqual(len(event_rows), 1)
        self.assertEqual(event_rows[0]["symbol"], "7203")
        self.assertEqual(event_rows[0]["breaker_reason"], "close_below_ma")
        self.assertEqual(event_rows[0]["action_taken"], "skip_entry")

        self.assertEqual(len(simulated_rows), 1)
        self.assertEqual(simulated_rows[0]["side"], "BUY")
        self.assertTrue(simulated_rows[0]["blocked_by_breaker"])
        self.assertEqual(simulated_rows[0]["blocker_reason"], "close_below_ma")
        self.assertEqual(simulated_rows[0]["would_open_or_close"], "open")

    def test_append_observation_rows_writes_header_and_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            event_path = os.path.join(temp_dir, "breaker_event_log.csv")
            simulated_path = os.path.join(temp_dir, "simulated_order_log.csv")

            append_observation_rows(
                event_path,
                BREAKER_EVENT_LOG_COLUMNS,
                [
                    {
                        "timestamp": "2026-03-09 09:00:00",
                        "symbol": "7203",
                        "side_candidate": "BUY",
                        "intended_qty": "",
                        "breaker_name": "market_breaker",
                        "breaker_reason": "close_below_ma",
                        "market_phase": "pre_market_batch",
                        "price_reference": 2500.0,
                        "volatility_reference": 0.02,
                        "spread_reference": "",
                        "action_taken": "skip_entry",
                    }
                ],
            )
            append_observation_rows(
                simulated_path,
                SIMULATED_ORDER_LOG_COLUMNS,
                [
                    {
                        "timestamp": "2026-03-09 09:00:00",
                        "symbol": "7203",
                        "side": "BUY",
                        "qty": "",
                        "order_type": "MARKET",
                        "intended_price": 2500.0,
                        "signal_score": 1.25,
                        "model_decision": "recommendation_candidate",
                        "blocked_by_breaker": True,
                        "blocker_reason": "close_below_ma",
                        "would_open_or_close": "open",
                    }
                ],
            )

            event_df = pd.read_csv(event_path, encoding="utf-8-sig")
            simulated_df = pd.read_csv(simulated_path, encoding="utf-8-sig")

            self.assertEqual(list(event_df.columns), BREAKER_EVENT_LOG_COLUMNS)
            self.assertEqual(list(simulated_df.columns), SIMULATED_ORDER_LOG_COLUMNS)
            self.assertEqual(str(event_df.iloc[0]["action_taken"]), "skip_entry")
            self.assertEqual(str(simulated_df.iloc[0]["blocker_reason"]), "close_below_ma")


if __name__ == "__main__":
    unittest.main(verbosity=2)
