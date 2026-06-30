'''
Author: fufeng
Description: 
Date: 2026-03-17 22:07:10
LastEditTime: 2026-03-17 23:48:40
FilePath: /quant_v5/core/datasource/eastmoney.py
'''
"""
EastMoney 直连客户端
BUG FIX: 原版 beg/end 直接传入带连字符日期 (YYYY-MM-DD)，
         东方财富 API 只接受 YYYYMMDD，导致返回空数据，
         进而回退到旧缓存（显示2025年12月底数据）。
"""
import pandas as pd
from .session import RetrySession
from .limiter import RateLimiter
from .market import MarketResolver

EM_KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


def _fmt_date(d: str) -> str:
    """任意日期格式 → YYYYMMDD（去掉连字符）"""
    return d.replace("-", "")


class EastMoneyClient:

    def __init__(self):
        self.session = RetrySession()
        self.limiter = RateLimiter()

    def kline(self, code: str, start: str, end: str) -> pd.DataFrame:
        secid = MarketResolver.secid(code)

        params = {
            "secid":   secid,
            "klt":     "101",
            "fqt":     "1",
            # BUG FIX: 格式化为 YYYYMMDD
            "beg":     _fmt_date(start),
            "end":     _fmt_date(end),
            "fields1": "f1,f2,f3,f4",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        }

        self.limiter.wait()
        r = self.session.get(EM_KLINE, params=params)

        try:
            data = r.json()
        except Exception:
            raise RuntimeError(f"EastMoney返回非JSON:\n{r.text[:200]}")

        klines = (data.get("data") or {}).get("klines")
        if not klines:
            return pd.DataFrame()

        rows = [k.split(",") for k in klines]
        df = pd.DataFrame(rows).iloc[:, :6]
        df.columns = ["date", "open", "close", "high", "low", "volume"]
        df["date"] = pd.to_datetime(df["date"])
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["code"] = code
        return df