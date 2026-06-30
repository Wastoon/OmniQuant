import pandas as pd


class DataValidator:
    REQUIRED_KLINE_COLUMNS = ("date", "open", "close", "high", "low")

    @staticmethod
    def check_empty(df):

        if df is None or df.empty:
            raise ValueError("数据为空")

    @staticmethod
    def check_dates(df):

        if "date" not in df.columns:
            return

        if df["date"].isnull().any():
            raise ValueError("日期缺失")

    @staticmethod
    def validate_kline(df):
        DataValidator.validate(df)

        missing = [c for c in DataValidator.REQUIRED_KLINE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"K线字段缺失: {','.join(missing)}")

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for col in ["open", "close", "high", "low", "volume"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        if out["date"].isnull().any():
            raise ValueError("日期缺失")
        if out["close"].isnull().all():
            raise ValueError("收盘价全部为空")

        out = out.dropna(subset=["date", "close"])
        out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

        bad_price = (out[["open", "close", "high", "low"]] <= 0).any(axis=1)
        if bad_price.any():
            out = out.loc[~bad_price].reset_index(drop=True)
        if out.empty:
            raise ValueError("有效价格数据为空")

        return out

    @staticmethod
    def validate(df):

        DataValidator.check_empty(df)

        DataValidator.check_dates(df)

        return df
