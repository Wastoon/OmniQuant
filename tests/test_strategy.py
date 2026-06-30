import unittest

import numpy as np
import pandas as pd

from core.factors import calc_drawdown, calc_technical
from core.strategy import backtest_strategy, generate_trade_advice


class StrategyAdviceTest(unittest.TestCase):
    def test_buy_advice_for_uptrend_with_good_score(self):
        prices = pd.Series(np.linspace(10, 18, 120))
        tech = calc_technical(prices)
        dd = calc_drawdown(prices.tolist())
        advice = generate_trade_advice(
            "stock",
            prices.tolist(),
            technical=tech,
            drawdown=dd,
            valuation={"pe_pct_5y": 20, "pb_pct_5y": 25},
            quant_score={"total_score": 82},
        )

        self.assertEqual(advice["action"], "buy")
        self.assertGreaterEqual(advice["suggested_position"], 50)
        self.assertGreater(advice["confidence"], 50)

    def test_reduce_advice_for_weak_downtrend(self):
        prices = pd.Series(np.linspace(20, 10, 120))
        tech = calc_technical(prices)
        dd = calc_drawdown(prices.tolist())
        advice = generate_trade_advice(
            "stock",
            prices.tolist(),
            technical=tech,
            drawdown=dd,
            valuation={"pe_pct_5y": 90, "pb_pct_5y": 85},
            quant_score={"total_score": 25},
        )

        self.assertEqual(advice["action"], "reduce")
        self.assertLessEqual(advice["suggested_position"], 15)
        self.assertTrue(advice["warnings"])


class BacktestStrategyTest(unittest.TestCase):
    def test_backtest_returns_stable_metrics(self):
        prices = np.concatenate([
            np.linspace(10, 14, 90),
            np.linspace(14, 12, 40),
            np.linspace(12, 20, 90),
        ])
        df = pd.DataFrame({"close": prices})
        result = backtest_strategy(df)

        self.assertIn("total_return_pct", result)
        self.assertIn("max_drawdown_pct", result)
        self.assertGreaterEqual(result["trade_count"], 1)
        self.assertEqual(len(result["equity_curve"]), len(df))

    def test_backtest_rejects_short_series(self):
        with self.assertRaises(ValueError):
            backtest_strategy(pd.DataFrame({"close": [1, 2, 3]}))


if __name__ == "__main__":
    unittest.main()
