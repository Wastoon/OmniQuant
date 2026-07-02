import pandas as pd
import numpy as np


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    机构级数据清洗（强制稳定）
    """

    # 1. 替换 inf
    df = df.replace([np.inf, -np.inf], np.nan)

    # 2. 可选：前后填充（避免指标断裂）
    # pandas 3.x 移除了 fillna(method=...)，改用等价的 ffill/bfill。
    df = df.ffill().bfill()

    # 3. 最终兜底 → None（JSON安全）
    df = df.where(pd.notnull(df), None)

    return df

def clean_for_json(obj):
    import numpy as np
    import pandas as pd

    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (pd.Timestamp,)):
        if pd.isna(obj):
            return None
        return obj.isoformat()
    elif obj is pd.NaT:
        return None
    return obj
    
