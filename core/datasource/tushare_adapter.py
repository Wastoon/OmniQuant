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
