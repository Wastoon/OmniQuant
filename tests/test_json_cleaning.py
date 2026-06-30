import json
import unittest

import numpy as np
import pandas as pd

from core.datasource.tools import clean_for_json


class JsonCleaningTest(unittest.TestCase):
    def test_clean_for_json_removes_non_compliant_float_values(self):
        payload = {
            "nan": float("nan"),
            "inf": np.float64(float("inf")),
            "ok": np.float64(1.23),
            "items": [np.int64(3), pd.NaT],
            "time": pd.Timestamp("2026-01-02"),
        }

        cleaned = clean_for_json(payload)
        encoded = json.dumps(cleaned, allow_nan=False)

        self.assertIn('"nan": null', encoded)
        self.assertIn('"inf": null', encoded)
        self.assertEqual(cleaned["ok"], 1.23)
        self.assertEqual(cleaned["items"], [3, None])


if __name__ == "__main__":
    unittest.main()
