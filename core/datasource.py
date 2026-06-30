"""
数据源适配器 v3.2 - AkShare 优先，Tushare 自动 Fallback
=========================================================

根因修复:
  RemoteDisconnected = 东方财富对无 User-Agent 的裸请求触发反爬，
  AkShare 内部 requests.get 没有设置任何 headers，第2次起连续请求被断开。

修复清单:
  FIX-1  自定义 Session + 完整浏览器 headers，绕过 AkShare 内部裸请求
  FIX-2  指数退避自动重试（1s/2s/3s），RemoteDisconnected 后重建 Session
  FIX-3  磁盘缓存（默认 6h TTL），财务数据 24h TTL，相同参数不重复请求
  FIX-4  限速器（请求间隔 ≥ 0.4s），避免触发东方财富反爬频率限制
  FIX-5  end_date 强制 cap 到今天，不传未来日期给接口
  FIX-6  Tushare adj_factor merge key 统一为 date
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi.encoders import jsonable_encoder

logger = logging.getLogger("datasource")

# ── 缓存配置 ──────────────────────────────────────────────────
CACHE_DIR = Path(os.getenv("QUANT_CACHE_DIR", str(Path.home() / ".quant_cache")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = int(os.getenv("QUANT_CACHE_TTL", str(6 * 3600)))   # 默认 6h
CACHE_TTL_FINANCIAL = 24 * 3600                                  # 财务数据 24h

# ── 限速 ──────────────────────────────────────────────────────
_MIN_INTERVAL = float(os.getenv("QUANT_MIN_INTERVAL", "0.4"))   # 秒
_last_req_ts: float = 0.0

# ── 东方财富接口 URL ──────────────────────────────────────────
_EM_KLINE  = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_EM_FUND   = "https://api.fund.eastmoney.com/f10/lsjz"

_ADJ   = {"qfq": "1", "hfq": "2", "": "0"}
_PERIOD = {"daily": "101", "weekly": "102", "monthly": "103"}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://finance.eastmoney.com/",
}


# ─────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────

def _sf(v, d=None):
    try:
        import math
        f = float(str(v).replace("%","").replace(",","").strip())
        return f if math.isfinite(f) else d
    except Exception:
        return d

def _today() -> str:
    return datetime.now().strftime("%Y%m%d")

def _ymd(d: str) -> str:
    """任意格式日期 → YYYYMMDD，并 cap 到今天 (FIX-5)"""
    return min(d.replace("-", ""), _today())

def _exchange(code: str) -> str:
    return "SH" if code[:1] in ("6", "5") else "SZ"

def _secid(code: str) -> str:
    return f"1.{code}" if _exchange(code) == "SH" else f"0.{code}"


# ─────────────────────────────────────────────────────────────
# 磁盘缓存 (FIX-3)
# ─────────────────────────────────────────────────────────────

def _ckey(*args, **kw) -> str:
    h = hashlib.md5(json.dumps([args, kw], sort_keys=True).encode()).hexdigest()[:12]
    return h

def _cload(key: str, ttl: int = CACHE_TTL) -> Optional[pd.DataFrame]:
    p = CACHE_DIR / f"{key}.parquet"
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > ttl:
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None

def _csave(key: str, df: pd.DataFrame):
    try:
        df.to_parquet(CACHE_DIR / f"{key}.parquet", index=False)
    except Exception as e:
        logger.debug("cache save failed: %s", e)


# ─────────────────────────────────────────────────────────────
# Session 管理 (FIX-1, FIX-2, FIX-4)
# ─────────────────────────────────────────────────────────────

_session: Optional[requests.Session] = None

def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    retry = Retry(total=3, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s

def _sess() -> requests.Session:
    global _session
    if _session is None:
        _session = _new_session()
    return _session

def _get(url: str, params: dict, timeout: int = 15, referer: str = "") -> requests.Response:
    """限速 + 断线重建 Session 的 GET (FIX-1, FIX-2, FIX-4)"""
    global _session, _last_req_ts
    wait = _MIN_INTERVAL - (time.time() - _last_req_ts)
    if wait > 0:
        time.sleep(wait)

    s = _sess()
    if referer:
        s.headers["Referer"] = referer

    for attempt in range(3):
        try:
            r = s.get(url, params=params, timeout=timeout)
            _last_req_ts = time.time()
            return r
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            logger.warning("连接断开 attempt=%d，重建 Session: %s", attempt+1, e)
            _session = _new_session()  # FIX-2: 重建 Session
            s = _session
            time.sleep(1.2 * (attempt + 1))

    raise requests.exceptions.ConnectionError("3 次重试后连接仍失败")


# ─────────────────────────────────────────────────────────────
# 东方财富直连（绕过 AkShare 裸请求）
# ─────────────────────────────────────────────────────────────

def _em_kline(code: str, start: str, end: str, adj: str = "qfq") -> pd.DataFrame:
    """股票 K 线直连，返回标准列 date/open/high/low/close/volume/amount/pct_chg"""
    ck = _ckey("kline", code, start, end, adj)
    hit = _cload(ck)
    if hit is not None:
        return hit

    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut":   "7eea3edcaed734bea9cbfc24409ed989",
        "klt":  _PERIOD.get("daily", "101"),
        "fqt":  _ADJ.get(adj, "1"),
        "secid": _secid(code),
        "beg":  _ymd(start),
        "end":  _ymd(end),
    }
    r = _get(_EM_KLINE, params)
    klines = (r.json().get("data") or {}).get("klines") or []
    if not klines:
        return pd.DataFrame()

    rows = [k.split(",") for k in klines]
    # fields2 固定顺序
    base_cols = ["date","open","close","high","low","volume","amount","amp","pct_chg","change","turnover"]
    ncols = len(rows[0])
    cols  = (base_cols + ["extra"])[:ncols]

    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open","high","low","close","volume","amount","pct_chg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["code"] = code

    out_cols = [c for c in ["date","code","open","high","low","close","volume","amount","pct_chg"] if c in df.columns]
    df = df[out_cols].sort_values("date").reset_index(drop=True)
    _csave(ck, df)
    return df


def _em_fund_nav(code: str, start: str, end: str) -> pd.DataFrame:
    """基金净值直连，返回标准列 date/nav/daily_return"""
    ck = _ckey("fnav", code, start, end)
    hit = _cload(ck)
    if hit is not None:
        return hit

    rows, page = [], 1
    # 日期转为 YYYY-MM-DD
    def fmt(d):
        d = d.replace("-","")
        return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d)==8 else d

    while True:
        params = {
            "fundCode":  code,
            "pageIndex": page,
            "pageSize":  2000,
            "startDate": fmt(start),
            "endDate":   fmt(end),
            "token":     "70786df93efd6b24d1c547f3d193dd4a",
        }
        try:
            r = _get(_EM_FUND, params, referer="https://fundf10.eastmoney.com/")
            data  = r.json()
            items = (data.get("Data") or {}).get("LSJZList") or []
            if not items:
                break
            rows.extend(items)
            if page * 2000 >= data.get("TotalCount", 0):
                break
            page += 1
            time.sleep(_MIN_INTERVAL)
        except Exception as e:
            logger.warning("基金净值 page=%d 失败: %s", page, e)
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.rename(columns={"FSRQ":"date","DWJZ":"nav","JZZZL":"daily_return"})
    df["date"] = pd.to_datetime(df["date"])
    df["nav"]  = pd.to_numeric(df.get("nav", pd.Series(dtype=float)), errors="coerce")
    df["daily_return"] = pd.to_numeric(df.get("daily_return", pd.Series(dtype=float)), errors="coerce").fillna(0)
    df = df[["date","nav","daily_return"]].dropna(subset=["nav"]).sort_values("date").reset_index(drop=True)
    _csave(ck, df)
    return df


# ─────────────────────────────────────────────────────────────
# DataUnavailableError
# ─────────────────────────────────────────────────────────────

class DataUnavailableError(Exception):
    pass


# ─────────────────────────────────────────────────────────────
# AkShare 适配器
# ─────────────────────────────────────────────────────────────

class AkShareAdapter:
    def __init__(self):
        try:
            import akshare as ak
            self._ak = ak
            self.available = True
            logger.info("AkShare v%s 就绪", ak.__version__)
        except ImportError:
            self.available = False
            logger.warning("AkShare 未安装")

    # K线：直接走直连，不用 AkShare 内部裸请求 (FIX-1)
    def stock_hist(self, code: str, start: str, end: str) -> pd.DataFrame:
        df = _em_kline(code, start, end, adj="qfq")
        if df.empty:
            raise ValueError(f"{code} 无数据，检查代码/日期范围")
        return df

    # 基金：AkShare 优先，失败用直连
    def fund_nav(self, code: str, period: str = "成立来") -> pd.DataFrame:
        try:
            df = self._ak.fund_open_fund_info_em(
                symbol=code, indicator="单位净值走势", period=period
            )
            rn = {"净值日期":"date","单位净值":"nav","日增长率":"daily_return"}
            df = df.rename(columns={k:v for k,v in rn.items() if k in df.columns})
            df["date"] = pd.to_datetime(df["date"])
            df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
            df["daily_return"] = df.get("daily_return", pd.Series(dtype=float)).apply(lambda x: _sf(x, 0))
            df = df.dropna(subset=["nav"]).sort_values("date").reset_index(drop=True)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning("AkShare fund_open_fund_info_em 失败，直连: %s", e)

        # 直连
        pmap = {"1月":30,"3月":90,"6月":180,"1年":365,"3年":1095,"5年":1825,"成立来":9999}
        days  = pmap.get(period, 9999)
        end   = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        df = _em_fund_nav(code, start, end)
        if df.empty:
            raise ValueError(f"基金 {code} 净值为空")
        return df

    def stock_info(self, code: str) -> dict:
        ck = _ckey("info", code)
        hit = _cload(ck, ttl=300)   # 5 分钟缓存（实时数据）

        if hit is not None and not hit.empty:
            hit = hit.replace([np.inf, -np.inf], np.nan)
            data = hit.iloc[0].to_dict()
            return jsonable_encoder(data)
        try:
            df = self._ak.stock_individual_info_em(symbol=code)
            result = {}
            for _, row in df.iterrows():
                k, v = str(row.iloc[0]), row.iloc[1]
                result[k] = v
                if "市盈率" in k:    result["pe_ttm"]     = _sf(v)
                elif "市净率" in k:  result["pb"]         = _sf(v)
                elif "总市值" in k:  result["market_cap"] = _sf(v)
                elif "所属行业" in k: result["industry"]   = str(v)
                elif "股票名称" in k: result["name"]       = str(v)
            if result:
                _csave(ck, pd.DataFrame([result]))
            return result
        except Exception as e:
            logger.warning("stock_info %s: %s", code, e)
            return {}

    def _cc(self, key: str, fn) -> pd.DataFrame:
        """带缓存的财务数据调用（24h TTL）"""
        hit = _cload(key, ttl=CACHE_TTL_FINANCIAL)
        if hit is not None:
            return hit
        try:
            df = fn()
            if df is not None and not df.empty:
                _csave(key, df)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            logger.warning("财务数据获取失败 %s: %s", key, e)
            return pd.DataFrame()

    def stock_balance_sheet(self, code: str) -> pd.DataFrame:
        return self._cc(f"bs_{code}", lambda: self._ak.stock_balance_sheet_by_report_em(symbol=code))

    def stock_income_sheet(self, code: str) -> pd.DataFrame:
        return self._cc(f"inc_{code}", lambda: self._ak.stock_profit_sheet_by_report_em(symbol=code))

    def stock_cashflow_sheet(self, code: str) -> pd.DataFrame:
        return self._cc(f"cf_{code}", lambda: self._ak.stock_cash_flow_sheet_by_report_em(symbol=code))

    def stock_financial_indicator(self, code: str) -> pd.DataFrame:
        return self._cc(f"fin_{code}", lambda: self._ak.stock_financial_analysis_indicator(symbol=code, start_year="2019"))

    def stock_dividend(self, code: str) -> pd.DataFrame:
        return self._cc(f"div_{code}", lambda: self._ak.stock_history_dividend_detail(symbol=code, indicator="分红"))

    def stock_fund_holdings(self, code: str, date: str) -> pd.DataFrame:
        return self._cc(f"inst_{code}_{date}", lambda: self._ak.stock_report_fund_hold(symbol=code, date=date))

    def stock_pe_pb_history(self, code: str) -> pd.DataFrame:
        return self._cc(f"pepb_{code}", lambda: self._ak.stock_a_lg_indicator(symbol=code))

    def search_stock(self, query: str) -> pd.DataFrame:
        try:
            df = self._ak.stock_info_a_code_name()
            return df[df["name"].str.contains(query, na=False)|df["code"].str.contains(query, na=False)].head(10)
        except Exception:
            return pd.DataFrame()

    def search_fund(self, query: str) -> pd.DataFrame:
        try:
            df = self._ak.fund_name_em()
            return df[df["基金简称"].str.contains(query, na=False)|df["基金代码"].str.contains(query, na=False)].head(10)
        except Exception:
            return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# Tushare 适配器
# ─────────────────────────────────────────────────────────────

class TushareAdapter:
    def __init__(self, token: Optional[str] = None):
        self._token = token or os.getenv("TUSHARE_TOKEN", "")
        self.available = False
        self._pro = None
        if self._token:
            try:
                import tushare as ts
                ts.set_token(self._token)
                self._pro = ts.pro_api()
                self.available = True
                logger.info("Tushare 就绪")
            except Exception as e:
                logger.warning("Tushare 初始化失败: %s", e)

    def _tscode(self, code: str) -> str:
        if "." in code:
            return code
        return f"{code}.{_exchange(code)}"

    def stock_hist(self, code: str, start: str, end: str) -> pd.DataFrame:
        # 先尝试直连（与 AkShare 共用，有缓存直接返回）
        df = _em_kline(code, start, end, adj="qfq")
        if not df.empty:
            return df
        # 真正的 Tushare fallback
        end_cap = _ymd(end)  # FIX-5
        df = self._pro.daily(
            ts_code=self._tscode(code),
            start_date=_ymd(start),
            end_date=end_cap,
            fields="trade_date,open,high,low,close,vol,amount,pct_chg",
        )
        df = df.rename(columns={"trade_date":"date","vol":"volume"})
        # 前复权 (FIX-6: adj merge key 统一)
        try:
            adj = self._pro.adj_factor(ts_code=self._tscode(code),
                                        start_date=_ymd(start), end_date=end_cap)
            if not adj.empty:
                adj = adj.rename(columns={"trade_date":"date"})   # FIX-6
                df  = df.merge(adj[["date","adj_factor"]], on="date", how="left")
                df["adj_factor"] = df["adj_factor"].fillna(1.0)
                for c in ["open","high","low","close"]:
                    df[c] = df[c] * df["adj_factor"]
        except Exception as e:
            logger.warning("adj_factor 跳过（可能需积分）: %s", e)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def fund_nav(self, code: str, period: str = "成立来") -> pd.DataFrame:
        pmap = {"1月":30,"3月":90,"6月":180,"1年":365,"3年":1095,"5年":1825,"成立来":9999}
        days = pmap.get(period, 9999)
        end  = _today()                    # FIX-5
        start= (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        df   = self._pro.fund_nav(ts_code=code+".OF", start_date=start, end_date=end,
                                   fields="nav_date,unit_nav,daily_return")
        df = df.rename(columns={"nav_date":"date","unit_nav":"nav"})
        df["date"] = pd.to_datetime(df["date"])
        df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
        df["daily_return"] = pd.to_numeric(df.get("daily_return", pd.Series(dtype=float)), errors="coerce").fillna(0)
        return df.dropna(subset=["nav"]).sort_values("date").reset_index(drop=True)

    def stock_info(self, code: str) -> dict:
        try:
            df = self._pro.daily_basic(ts_code=self._tscode(code), trade_date="",
                                        fields="ts_code,pe_ttm,pb,ps_ttm,total_mv")
            if not df.empty:
                row = df.iloc[0]
                return {"pe_ttm":_sf(row.get("pe_ttm")),"pb":_sf(row.get("pb")),"market_cap":_sf(row.get("total_mv"))}
        except Exception:
            pass
        return {}

    def stock_balance_sheet(self, code: str) -> pd.DataFrame:
        return self._pro.balancesheet(ts_code=self._tscode(code),
            fields="end_date,accounts_payable,adv_receipts,contract_liab,total_liab")

    def stock_income_sheet(self, code: str) -> pd.DataFrame:
        return self._pro.income(ts_code=self._tscode(code),
            fields="end_date,revenue,operate_profit,n_income,rd_exp,gross_profit")

    def stock_cashflow_sheet(self, code: str) -> pd.DataFrame:
        return self._pro.cashflow(ts_code=self._tscode(code),
            fields="end_date,n_cashflow_act,c_pay_acq_const_fiolta")

    def stock_financial_indicator(self, code: str) -> pd.DataFrame:
        return self._pro.fina_indicator(ts_code=self._tscode(code),
            fields="end_date,roe,roa,netprofit_margin,grossprofit_margin,debt_to_assets,current_ratio,eps,bps")

    def stock_dividend(self, code: str) -> pd.DataFrame:
        return self._pro.dividend(ts_code=self._tscode(code), fields="end_date,cash_div_tax")

    def stock_fund_holdings(self, code: str, date: str) -> pd.DataFrame:
        return self._pro.fund_portfolio(ts_code=self._tscode(code), period=date[:6], fields="symbol,mkv,amount")

    def stock_pe_pb_history(self, code: str) -> pd.DataFrame:
        end   = _today()
        start = (datetime.now() - timedelta(days=365*5)).strftime("%Y%m%d")
        df = self._pro.daily_basic(ts_code=self._tscode(code), start_date=start, end_date=end,
                                    fields="trade_date,pe_ttm,pb")
        return df.rename(columns={"trade_date":"date"}).sort_values("date")

    def search_stock(self, query: str) -> pd.DataFrame:
        try:
            df = self._pro.stock_basic(fields="ts_code,symbol,name")
            return df[df["name"].str.contains(query,na=False)|df["symbol"].str.contains(query,na=False)].head(10).rename(columns={"symbol":"code"})
        except Exception:
            return pd.DataFrame()

    def search_fund(self, query: str) -> pd.DataFrame:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 统一门面
# ─────────────────────────────────────────────────────────────

class DataSource:
    """AkShare 优先，Tushare fallback，底层 K 线走自定义 Session + 缓存。"""

    def __init__(self, ts_token: Optional[str] = None):
        self.ak = AkShareAdapter()
        self.ts = TushareAdapter(token=ts_token)
        self._log: dict = {}
        if not self.ak.available and not self.ts.available:
            raise RuntimeError("AkShare 和 Tushare 均不可用，请 pip install akshare")
        logger.info("DataSource 就绪 | 主源: %s | 缓存: %s",
                    "AkShare" if self.ak.available else "Tushare", CACHE_DIR)

    @property
    def status(self) -> dict:
        return {
            "akshare":        self.ak.available,
            "tushare":        self.ts.available,
            "primary":        "akshare" if self.ak.available else "tushare",
            "cache_dir":      str(CACHE_DIR),
            "cache_ttl_hours": CACHE_TTL // 3600,
        }

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

    def stock_hist(self, code: str, start: str, end: str) -> pd.DataFrame:
        return self._call("stock_hist", code, start, end)

    def fund_nav(self, code: str, period: str = "成立来") -> pd.DataFrame:
        return self._call("fund_nav", code, period)

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