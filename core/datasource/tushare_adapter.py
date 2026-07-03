import math
from datetime import datetime, timedelta

import pandas as pd


def _sf(v, d=None):
    try:
        f = float(v)
        return f if math.isfinite(f) else d
    except Exception:
        return d


class TushareAdapter:

    def __init__(self, token=None):

        self.available = False

        if token:

            import tushare as ts

            ts.set_token(token)

            self.pro = ts.pro_api()

            self.available = True

    def stock_hist(self, code, start, end):

        if not self.available:
            return None

        ts_code = f"{code}.SH" if code.startswith("6") else f"{code}.SZ"

        df = self.pro.daily(
            ts_code=ts_code,
            start_date=start.replace("-", ""),
            end_date=end.replace("-", "")
        )

        return df

    def _tscode(self, code):
        if "." in str(code):
            return code
        return f"{code}.SH" if str(code).startswith("6") else f"{code}.SZ"

    def stock_info(self, code: str) -> dict:
        if not self.available:
            return {}
        df = self.pro.daily_basic(
            ts_code=self._tscode(code),
            trade_date="",
            fields="ts_code,trade_date,pe_ttm,pb,ps_ttm,total_mv,circ_mv",
        )
        if df is None or df.empty:
            return {}
        row = df.sort_values("trade_date").iloc[-1] if "trade_date" in df.columns else df.iloc[0]
        return {
            "pe_ttm": _sf(row.get("pe_ttm")),
            "pb": _sf(row.get("pb")),
            "ps_ttm": _sf(row.get("ps_ttm")),
            "market_cap": _sf(row.get("total_mv")),
            "circ_mv": _sf(row.get("circ_mv")),
        }

    def stock_pe_pb_history(self, code: str) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=365 * 5 + 30)).strftime("%Y%m%d")
        df = self.pro.daily_basic(
            ts_code=self._tscode(code),
            start_date=start,
            end_date=end,
            fields="trade_date,pe_ttm,pb",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return df.rename(columns={"trade_date": "date", "pe_ttm": "pe"}).sort_values("date").reset_index(drop=True)

    def stock_financial_indicator(self, code: str) -> pd.DataFrame:
        return self.pro.fina_indicator(
            ts_code=self._tscode(code),
            fields="end_date,roe,roa,netprofit_margin,grossprofit_margin,debt_to_assets,current_ratio,eps,bps",
        )

    def stock_balance_sheet(self, code: str) -> pd.DataFrame:
        return self.pro.balancesheet(
            ts_code=self._tscode(code),
            fields="end_date,accounts_payable,adv_receipts,contract_liab,total_liab,intangible_assets,total_assets",
        )

    def stock_income_sheet(self, code: str) -> pd.DataFrame:
        return self.pro.income(
            ts_code=self._tscode(code),
            fields="end_date,revenue,operate_profit,n_income,rd_exp,gross_profit",
        )

    def stock_cashflow_sheet(self, code: str) -> pd.DataFrame:
        return self.pro.cashflow(
            ts_code=self._tscode(code),
            fields="end_date,n_cashflow_act,c_pay_acq_const_fiolta",
        )

    def stock_dividend(self, code: str) -> pd.DataFrame:
        return self.pro.dividend(ts_code=self._tscode(code), fields="end_date,cash_div_tax")
