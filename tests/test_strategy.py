import unittest

import numpy as np
import pandas as pd

from core.factors import calc_drawdown, calc_technical
from core.strategy import (
    backtest_strategy,
    calc_ema_trailing_strategy,
    calc_startup_strategy,
    generate_trade_advice,
    summarize_strategy_result,
)


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


class EmaTrailingStrategyTest(unittest.TestCase):
    def test_ema_trailing_accepts_close_only_data(self):
        prices = np.concatenate([
            np.linspace(10, 11, 40),
            np.linspace(11, 15, 40),
            np.linspace(15, 13, 30),
        ])
        result = calc_ema_trailing_strategy(pd.DataFrame({"close": prices}))

        self.assertEqual(len(result["equity_curve"]), len(prices))
        self.assertEqual(len(result["buy_signals"]), len(prices))
        self.assertIn("params", result)

    def test_ema_trailing_scales_position_down_when_volatility_is_high(self):
        low_vol = pd.DataFrame({
            "close": np.concatenate([np.linspace(10, 11, 40), np.linspace(11, 20, 40)]),
            "high": np.concatenate([np.linspace(10, 11, 40), np.linspace(11, 20, 40)]) * 1.03,
            "low": np.concatenate([np.linspace(10, 11, 40), np.linspace(11, 20, 40)]) * 0.97,
        })
        high_vol_close = np.concatenate([np.linspace(10, 11, 40), np.linspace(11, 20, 40)])
        high_vol = pd.DataFrame({
            "close": high_vol_close,
            "high": high_vol_close * 1.08,
            "low": high_vol_close * 0.92,
        })

        low_result = calc_ema_trailing_strategy(low_vol, max_atr_pct=1.0, max_entry_extension=0.5, min_position=0.05)
        high_result = calc_ema_trailing_strategy(high_vol, max_atr_pct=1.0, max_entry_extension=0.5, min_position=0.05)
        low_buy = next(t for t in low_result["trades"] if t["action"] == "buy")
        high_buy = next(t for t in high_result["trades"] if t["action"] == "buy")

        self.assertGreater(low_buy["position_pct"], high_buy["position_pct"])
        self.assertLessEqual(high_buy["position_pct"], 90)

    def test_ema_trailing_uses_fund_defaults_to_reduce_turnover(self):
        prices = np.concatenate([
            np.linspace(10, 11, 50),
            np.linspace(11, 13, 25),
            np.linspace(13, 12.2, 8),
            np.linspace(12.2, 13.2, 18),
            np.linspace(13.2, 11.2, 35),
        ])
        result = calc_ema_trailing_strategy(pd.DataFrame({"close": prices}), asset_type="fund")
        sells = [t for t in result["trades"] if t["action"] == "sell"]

        self.assertEqual(result["params"]["min_holding_days"], 30)
        self.assertEqual(result["params"]["cooldown_days"], 10)
        self.assertEqual(result["params"]["sell_confirm_days"], 2)
        self.assertTrue(all(t["holding_days"] >= 30 or t["reason"] == "hard_stop" for t in sells))

    def test_ema_trailing_fund_can_enter_slow_uptrend(self):
        prices = np.linspace(10, 14, 160)
        result = calc_ema_trailing_strategy(pd.DataFrame({"close": prices}), asset_type="fund")
        buys = [t for t in result["trades"] if t["action"] == "buy"]

        self.assertTrue(buys)
        self.assertGreaterEqual(buys[0]["position_pct"], 35)

    def test_ema_trailing_fund_uses_stop_loss_floor(self):
        prices = np.linspace(10, 14, 160)
        result = calc_ema_trailing_strategy(pd.DataFrame({"close": prices}), asset_type="fund")
        buy = next(t for t in result["trades"] if t["action"] == "buy")

        self.assertEqual(result["params"]["min_stop_loss_pct"], 0.08)
        self.assertLessEqual(buy["stop_price"], round(buy["price"] * 0.92, 3))

    def test_ema_trailing_applies_fee_and_slippage(self):
        prices = np.concatenate([np.linspace(10, 11, 40), np.linspace(11, 14, 40)])
        result = calc_ema_trailing_strategy(
            pd.DataFrame({"close": prices}),
            fee_rate=0.01,
            slippage_rate=0.02,
            volume_confirm=False,
        )
        buy = next(t for t in result["trades"] if t["action"] == "buy")

        self.assertGreater(buy["exec_price"], buy["price"])
        self.assertEqual(result["params"]["fee_rate"], 0.01)

    def test_ema_trailing_volume_confirmation_can_filter_stock_breakout(self):
        prices = np.concatenate([np.linspace(10, 11, 40), np.linspace(11, 14, 40)])
        low_volume = np.ones_like(prices) * 100
        result = calc_ema_trailing_strategy(
            pd.DataFrame({"close": prices, "volume": low_volume}),
            asset_type="stock",
            volume_confirm=True,
        )

        self.assertFalse(any(t["action"] == "buy" for t in result["trades"]))

    def test_ema_trailing_stock_uses_stop_loss_floor(self):
        prices = np.linspace(10, 20, 120)
        stock_df = pd.DataFrame({"close": prices, "high": prices * 1.03, "low": prices * 0.97})
        result = calc_ema_trailing_strategy(
            stock_df,
            asset_type="stock",
            volume_confirm=False,
            use_long_trend_filter=False,
            max_entry_extension=0.5,
        )
        buy = next(t for t in result["trades"] if t["action"] == "buy")

        self.assertEqual(result["params"]["min_stop_loss_pct"], 0.04)
        self.assertLessEqual(buy["stop_price"], round(buy["price"] * 0.96, 3))

    def test_ema_trailing_stock_filters_extreme_volatility(self):
        prices = np.linspace(10, 20, 120)
        high_vol = pd.DataFrame({
            "close": prices,
            "high": prices * 1.12,
            "low": prices * 0.88,
        })
        result = calc_ema_trailing_strategy(
            high_vol,
            asset_type="stock",
            volume_confirm=False,
            use_long_trend_filter=False,
            max_atr_pct=0.08,
        )

        self.assertFalse(any(t["action"] == "buy" for t in result["trades"]))

    def test_summarize_strategy_result_returns_regression_metrics(self):
        prices = np.linspace(10, 16, 120)
        result = calc_ema_trailing_strategy(
            pd.DataFrame({"close": prices, "high": prices * 1.03, "low": prices * 0.97}),
            asset_type="stock",
            volume_confirm=False,
            use_long_trend_filter=False,
            max_entry_extension=0.5,
        )
        summary = summarize_strategy_result(result, prices)

        self.assertIn("total_return_pct", summary)
        self.assertIn("buy_hold_return_pct", summary)
        self.assertIn("max_drawdown_pct", summary)
        self.assertGreaterEqual(summary["trade_count"], 1)


class StartupStrategyTest(unittest.TestCase):
    def test_startup_strategy_returns_qfq_indicators_and_signal_layers(self):
        prices = np.concatenate([
            np.linspace(10, 12, 70),
            np.linspace(12, 10.8, 20),
            np.linspace(10.8, 15, 80),
        ])
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=len(prices), freq="D"),
            "close": prices,
            "volume": np.full(len(prices), 1000.0),
        })

        result = calc_startup_strategy(df, volume_confirm=False)

        self.assertEqual(result["method"], "qfq_ma_ene_dma_macd_startup")
        self.assertEqual(len(result["buy_signals"]), len(prices))
        self.assertEqual(len(result["sell_signals"]), len(prices))
        self.assertEqual(len(result["equity_curve"]), len(prices))
        for name in ("ma5", "ma10", "ma20", "ma30", "ene_mid", "ene_upper",
                     "ene_lower", "dma_ddd", "dma_ama", "macd_dif",
                     "macd_dea", "macd_hist"):
            self.assertEqual(len(result["indicators"][name]), len(prices))

    def test_startup_strategy_uses_dma_definition_and_reports_sell_reason(self):
        prices = np.concatenate([
            np.linspace(10, 14, 100),
            np.linspace(14, 12.2, 35),
        ])
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=len(prices), freq="D"),
            "close": prices,
            "volume": np.full(len(prices), 1000.0),
        })

        result = calc_startup_strategy(
            df,
            volume_confirm=False,
            trailing_stop_pct=0.03,
            confirmation_window=30,
        )
        indicators = result["indicators"]
        for index in (60, 90, 120):
            expected = (
                pd.Series(prices).rolling(10).mean().iloc[index]
                - pd.Series(prices).rolling(50).mean().iloc[index]
            )
            self.assertAlmostEqual(indicators["dma_ddd"][index], expected, places=5)
        sells = [trade for trade in result["trades"] if trade["action"] == "sell"]
        self.assertTrue(all(trade.get("reason") for trade in sells))


if __name__ == "__main__":
    unittest.main()
