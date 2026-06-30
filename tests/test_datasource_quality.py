import unittest

import pandas as pd

from core.datasource.datasource import normalize_kline
from core.datasource.validator import DataValidator


class DataQualityTest(unittest.TestCase):
    def test_normalize_kline_handles_akshare_columns(self):
        raw = pd.DataFrame({
            "日期": ["2026-01-02", "2026-01-03"],
            "开盘": [10, 11],
            "收盘": [11, 12],
            "最高": [11.5, 12.5],
            "最低": [9.8, 10.8],
            "成交量": [1000, 1200],
        })

        df = normalize_kline(raw)
        checked = DataValidator.validate_kline(df)

        self.assertEqual(list(checked["close"]), [11, 12])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(checked["date"]))

    def test_validator_drops_duplicate_and_bad_prices(self):
        raw = pd.DataFrame({
            "date": ["2026-01-02", "2026-01-02", "2026-01-03"],
            "open": [10, 10.5, 0],
            "close": [11, 11.2, 12],
            "high": [11.5, 11.8, 12.5],
            "low": [9.8, 10.1, 10.8],
            "volume": [1000, 1100, 1200],
        })

        checked = DataValidator.validate_kline(raw)

        self.assertEqual(len(checked), 1)
        self.assertEqual(float(checked.iloc[0]["close"]), 11.2)

    def test_validator_rejects_empty_data(self):
        with self.assertRaises(ValueError):
            DataValidator.validate_kline(pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
