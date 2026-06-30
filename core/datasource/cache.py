import time
import pandas as pd
from pathlib import Path
from core.datasource.config import CACHE_DIR, CACHE_TTL


class CacheManager:

    def __init__(self):

        self.dir = CACHE_DIR

    def path(self, key):

        return self.dir / f"{key}.parquet"

    def load(self, key):

        p = self.path(key)

        if not p.exists():
            return None

        if time.time() - p.stat().st_mtime > CACHE_TTL:
            return None

        return pd.read_parquet(p)

    def load_stale(self, key):

        p = self.path(key)

        if not p.exists():
            return None

        return pd.read_parquet(p)

    def save(self, key, df):

        p = self.path(key)

        df.to_parquet(p, index=False)
