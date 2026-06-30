'''
Author: fufeng
Description: 
Date: 2026-03-16 23:31:38
LastEditTime: 2026-03-17 00:35:33
FilePath: /quant_v3/core/datasource/datasource.py
'''
import logging
import time
import os
import pandas as pd
import numpy as np


try:
    from .eastmoney import EastMoneyClient
except Exception as e:
    EastMoneyClient = None
    _EASTMONEY_IMPORT_ERROR = e
else:
    _EASTMONEY_IMPORT_ERROR = None
from .akshare_adapter import AkShareAdapter
from .tushare_adapter import TushareAdapter
from .cache import CacheManager
from .validator import DataValidator
from .config import CACHE_DIR, CACHE_TTL

logger = logging.getLogger("datasource")

def normalize_kline(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # ── 日期列统一 ──
    if "date" not in df.columns:
        for col in ["日期", "trade_date", "datetime"]:
            if col in df.columns:
                df["date"] = df[col]
                break

    # 转 datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ── 价格列统一 ──
    rename_map = {
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "vol": "volume",
    }
    df.rename(columns=rename_map, inplace=True)

    if "date" in df.columns:
        as_text = df["date"].astype(str)
        if as_text.str.fullmatch(r"\d{8}").any():
            df["date"] = pd.to_datetime(as_text, format="%Y%m%d", errors="coerce")

    # 确保必要列存在
    required = ["date", "open", "close", "high", "low"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # 排序
    df = df.sort_values("date").reset_index(drop=True)

    return df

class DataUnavailableError(Exception):
    pass

class DataSource:

    def __init__(self, ts_token=None):

        self.em = EastMoneyClient() if EastMoneyClient else None

        self.ak = AkShareAdapter()

        self.ts = TushareAdapter(ts_token)

        self._log: dict = {}
        self.cache = CacheManager()

        if not self.ak.available and not self.ts.available:
            raise RuntimeError("AkShare 和 Tushare 均不可用，请 pip install akshare")
        logger.info("DataSource 就绪 | 主源: %s | 缓存: %s",
                    "AkShare" if self.ak.available else "Tushare", CACHE_DIR)
        if self.em is None:
            logger.warning("EastMoney 备用源不可用: %s", _EASTMONEY_IMPORT_ERROR)
        
    @property
    def status(self) -> dict:
        return {
            "akshare":        self.ak.available,
            "tushare":        self.ts.available,
            "primary":        "akshare" if self.ak.available else "tushare",
            "cache_dir":      str(CACHE_DIR),
            "cache_ttl_hours": CACHE_TTL // 3600,
        }
        
    def stock_hist(self, code, start, end):
        key = f"kline_{code}_{start}_{end}_D_v1"

        cached = self.cache.load(key)
        if cached is not None:
            self._log["stock_hist"] = "cache"
            return cached

        try:
            # 一级：AkShare + Tushare
            df = self._call("stock_hist", code, start, end)
            df = normalize_kline(df)
            df = DataValidator.validate_kline(df)
            self.cache.save(key, df)
            return df

        except Exception as e1:
            logger.warning("主数据源失败: %s", e1)

            # 二级：EastMoney 兜底
            try:
                if self.em is None:
                    raise RuntimeError(f"EastMoney client unavailable: {_EASTMONEY_IMPORT_ERROR}")
                df = self.em.kline(code, start, end)
                df = DataValidator.validate_kline(df)
                self._log["stock_hist"] = "eastmoney"
                self.cache.save(key, df)
                return df
            except Exception as e2:
                logger.error("EastMoney 也失败: %s", e2)

                # 三级：返回旧缓存（关键）
                stale = self.cache.load_stale(key)
                if stale is not None:
                    logger.warning("使用 stale cache: %s", key)
                    self._log["stock_hist"] = "stale_cache"
                    return stale

                raise DataUnavailableError(
                    f"所有数据源失败:\nAk/Tu: {e1}\nEM: {e2}"
                )

    def source_log(self) -> dict:
        return dict(self._log)
    
    def _call(self, method: str, *args, **kwargs):
        errors = []
        for adapter, name in [(self.ak, "AkShare"), (self.ts, "Tushare")]:
            if not adapter.available:
                continue
            if not hasattr(adapter, method):
                continue
            try:
                result = getattr(adapter, method)(*args, **kwargs)
                self._log[method] = name.lower()
                return result
            except Exception as e:
                errors.append(f"{name}.{method}: {e}")
                logger.debug("%s.%s 失败: %s", name, method, e)
        raise DataUnavailableError(
            f"方法 {method} 两个数据源均失败:\n" + "\n".join(errors)
        )

    # def fund_nav(self, code: str, period: str = "成立来") -> pd.DataFrame:
    #     return self._call("fund_nav", code, period)

    def fund_nav(self, code: str, period: str = "成立来") -> pd.DataFrame:
        """
        机构级基金净值获取（多源 fallback + 强校验）
        """
        key = f"fund_nav_{code}_{period}_v1"
        cached = self.cache.load(key)
        if cached is not None:
            self._log["fund_nav"] = "cache"
            return cached

        errors = []

        # ───── 1️⃣ AkShare 主源 ─────
        try:
            df = self.ak.ak.fund_open_fund_info_em(symbol=code)

            if df is not None and not df.empty:
                # 标准化字段
                col_map = {
                    "净值日期": "date",
                    "单位净值": "nav",
                    "累计净值": "acc_nav",
                }
                df = df.rename(columns=col_map)

                if "date" in df.columns and "nav" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")

                    df = df.sort_values("date").dropna(subset=["date", "nav"])

                    if len(df) > 10:
                        out = df[["date", "nav"]]
                        self.cache.save(key, out)
                        self._log["fund_nav"] = "akshare"
                        return out

            errors.append("akshare empty")

        except Exception as e:
            errors.append(f"akshare error: {e}")

        # ───── 2️⃣ EastMoney 直连（强烈推荐） ─────
        try:
            import requests

            url = f"https://api.fund.eastmoney.com/f10/lsjz"
            params = {
                "fundCode": code,
                "pageIndex": 1,
                "pageSize": 1000,
            }

            r = requests.get(url, params=params, timeout=5)
            data = r.json()

            if "Data" in data and "LSJZList" in data["Data"]:
                rows = data["Data"]["LSJZList"]

                df = pd.DataFrame(rows)

                df["date"] = pd.to_datetime(df["FSRQ"], errors="coerce")
                df["nav"] = pd.to_numeric(df["DWJZ"], errors="coerce")

                df = df.sort_values("date").dropna(subset=["date", "nav"])

                if len(df) > 10:
                    out = df[["date", "nav"]]
                    self.cache.save(key, out)
                    self._log["fund_nav"] = "eastmoney"
                    return out

            errors.append("eastmoney empty")

        except Exception as e:
            errors.append(f"eastmoney error: {e}")

        # ───── 3️⃣ 全失败 ─────
        stale = self.cache.load_stale(key)
        if stale is not None:
            self._log["fund_nav"] = "stale_cache"
            return stale

        raise DataUnavailableError(f"fund_nav 失败: {errors}")

    def stock_quote(self, code: str) -> dict:
        errors = []

        try:
            quote = self._call("stock_quote", code)
            if quote.get("price") is not None:
                self._log["stock_quote"] = self._log.get("stock_quote", "akshare")
                return quote
        except Exception as e:
            errors.append(f"adapter: {e}")

        try:
            end = pd.Timestamp.today().strftime("%Y-%m-%d")
            start = (pd.Timestamp.today() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
            hist = self.stock_hist(code, start, end)
            row = hist.iloc[-1]
            self._log["stock_quote"] = "hist_fallback"
            return {
                "code": code,
                "name": code,
                "price": float(row["close"]),
                "change_pct": None,
                "change": None,
                "volume": float(row["volume"]) if "volume" in hist.columns else None,
                "open": float(row["open"]) if "open" in hist.columns else None,
                "high": float(row["high"]) if "high" in hist.columns else None,
                "low": float(row["low"]) if "low" in hist.columns else None,
                "date": row["date"].strftime("%Y-%m-%d"),
            }
        except Exception as e:
            errors.append(f"hist fallback: {e}")

        raise DataUnavailableError(f"stock_quote 失败: {errors}")

    def market_price(self, code: str, asset_type: str = "stock") -> dict:
        if asset_type == "fund":
            df = self.fund_nav(code)
            row = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            nav = float(row["nav"])
            prev_nav = float(prev["nav"]) if prev is not None else None
            return {
                "code": code,
                "type": "fund",
                "price": nav,
                "nav": nav,
                "date": row["date"].strftime("%Y-%m-%d"),
                "change_pct": ((nav / prev_nav - 1) * 100) if prev_nav else None,
                "source": self._log.get("fund_nav"),
            }

        quote = self.stock_quote(code)
        quote["type"] = "stock"
        quote["source"] = self._log.get("stock_quote")
        return quote

    def stock_info(self, code: str) -> dict:
        try:
            return self._call("stock_info", code)
        except DataUnavailableError:
            return {}

    def stock_balance_sheet(self, code: str) -> pd.DataFrame:
        return self._call("stock_balance_sheet", code)

    def stock_income_sheet(self, code: str) -> pd.DataFrame:
        return self._call("stock_income_sheet", code)

    def stock_cashflow_sheet(self, code: str) -> pd.DataFrame:
        return self._call("stock_cashflow_sheet", code)

    def stock_financial_indicator(self, code: str) -> pd.DataFrame:
        return self._call("stock_financial_indicator", code)

    def stock_dividend(self, code: str) -> pd.DataFrame:
        return self._call("stock_dividend", code)

    def stock_fund_holdings(self, code: str, date: str) -> pd.DataFrame:
        return self._call("stock_fund_holdings", code, date)

    def stock_pe_pb_history(self, code: str) -> pd.DataFrame:
        return self._call("stock_pe_pb_history", code)

    def search_stock(self, query: str) -> pd.DataFrame:
        try: return self._call("search_stock", query)
        except DataUnavailableError: return pd.DataFrame()

    def search_fund(self, query: str) -> pd.DataFrame:
        try: return self._call("search_fund", query)
        except DataUnavailableError: return pd.DataFrame()

    def source_log(self) -> dict:
        return dict(self._log)

    def cache_info(self) -> dict:
        files = list(CACHE_DIR.glob("*.parquet"))
        return {
            "files":     len(files),
            "total_mb":  round(sum(f.stat().st_size for f in files) / 1024 / 1024, 2),
            "cache_dir": str(CACHE_DIR),
        }

    def clear_cache(self, older_than_hours: int = 0) -> int:
        n, cutoff = 0, time.time() - older_than_hours * 3600
        for f in CACHE_DIR.glob("*.parquet"):
            if f.stat().st_mtime < cutoff:
                f.unlink(); n += 1
        return n
