"""
AkShare 数据适配器 — 补全所有缺失方法
修复: 原版只有 stock_info，缺少所有财务/行情/搜索方法导致
      PE/PB、量化因子、行业分析全部返回空。
"""
import logging
import pandas as pd

logger = logging.getLogger("akshare_adapter")


def _sf(v, d=None):
    try:
        import math
        f = float(v)
        return f if math.isfinite(f) else d
    except Exception:
        return d


class AkShareAdapter:

    def __init__(self):
        try:
            import akshare as ak
            self.ak = ak
            self.available = True
            logger.info("AkShare v%s 就绪", ak.__version__)
        except ImportError:
            self.available = False
            logger.warning("AkShare 未安装")

    # ── 行情 ─────────────────────────────────────────────────────

    def stock_hist(self, code: str, start: str, end: str) -> pd.DataFrame:
        """
        股票日K线（前复权）
        BUG FIX: AkShare 需要 YYYYMMDD 格式，去掉连字符
        """
        start_clean = start.replace("-", "")
        end_clean   = end.replace("-", "")
        df = self.ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_clean,
            end_date=end_clean,
            adjust="qfq",
        )
        return df

    def stock_quote(self, code: str) -> dict:
        df = self.ak.stock_zh_a_spot_em()
        row = df[df["代码"].astype(str) == str(code)]
        if row.empty:
            raise ValueError(f"未找到股票实时行情: {code}")
        r = row.iloc[0]
        return {
            "code": str(r.get("代码", code)),
            "name": str(r.get("名称", code)),
            "price": _sf(r.get("最新价")),
            "change_pct": _sf(r.get("涨跌幅")),
            "change": _sf(r.get("涨跌额")),
            "volume": _sf(r.get("成交量")),
            "amount": _sf(r.get("成交额")),
            "open": _sf(r.get("今开")),
            "high": _sf(r.get("最高")),
            "low": _sf(r.get("最低")),
            "prev_close": _sf(r.get("昨收")),
            "pe_ttm": _sf(r.get("市盈率-动态")),
            "pb": _sf(r.get("市净率")),
        }

    # ── 基本信息 ──────────────────────────────────────────────────

    def stock_info(self, code: str) -> dict:
        df = self.ak.stock_individual_info_em(symbol=code)
        result = {}
        for _, r in df.iterrows():
            k = str(r.iloc[0])
            v = r.iloc[1]
            result[k] = v
            if "市盈率" in k:    result["pe_ttm"]     = _sf(v)
            elif "市净率" in k:  result["pb"]         = _sf(v)
            elif "总市值" in k:  result["market_cap"] = _sf(v)
            elif "所属行业" in k: result["industry"]   = str(v)
            elif "股票名称" in k: result["name"]       = str(v)
        return result

    # ── PE/PB 历史估值 ────────────────────────────────────────────

    def stock_pe_pb_history(self, code: str) -> pd.DataFrame:
        """
        PE/PB 历史百分位数据
        使用 stock_a_lg_indicator（龙虎榜历史估值）
        """
        df = self.ak.stock_a_lg_indicator(symbol=code)
        return df

    # ── 财务报表 ──────────────────────────────────────────────────

    def stock_financial_indicator(self, code: str) -> pd.DataFrame:
        """主要财务指标（ROE / 毛利率 / EPS 等）"""
        return self.ak.stock_financial_analysis_indicator(
            symbol=code, start_year="2019"
        )

    def stock_balance_sheet(self, code: str) -> pd.DataFrame:
        """资产负债表（无息负债比例计算用）"""
        return self.ak.stock_balance_sheet_by_report_em(symbol=code)

    def stock_income_sheet(self, code: str) -> pd.DataFrame:
        """利润表（研发费用等）"""
        return self.ak.stock_profit_sheet_by_report_em(symbol=code)

    def stock_cashflow_sheet(self, code: str) -> pd.DataFrame:
        """现金流量表（FCF 计算用）"""
        return self.ak.stock_cash_flow_sheet_by_report_em(symbol=code)

    def stock_dividend(self, code: str) -> pd.DataFrame:
        """分红历史（股息支付率计算用）"""
        return self.ak.stock_history_dividend_detail(
            symbol=code, indicator="分红"
        )

    def stock_fund_holdings(self, code: str, date: str) -> pd.DataFrame:
        """基金持仓（机构持仓分析用）"""
        return self.ak.stock_report_fund_hold(symbol=code, date=date)

    # ── 搜索 ──────────────────────────────────────────────────────

    def search_stock(self, query: str) -> pd.DataFrame:
        df = self.ak.stock_info_a_code_name()
        return df[
            df["name"].str.contains(query, na=False) |
            df["code"].str.contains(query, na=False)
        ].head(10)

    def search_fund(self, query: str) -> pd.DataFrame:
        df = self.ak.fund_name_em()
        return df[
            df["基金简称"].str.contains(query, na=False) |
            df["基金代码"].str.contains(query, na=False)
        ].head(10)
