'''
Author: fufeng
Description: 
Date: 2026-03-16 23:28:53
LastEditTime: 2026-03-16 23:51:34
FilePath: /quant_v3/core/datasource/config.py
'''
import os
from pathlib import Path

CACHE_DIR = Path(os.getenv("QUANT_CACHE_DIR", str(Path.home() / ".quant_cache")))
CACHE_DIR.mkdir(exist_ok=True, parents=True)

CACHE_TTL = 6 * 3600

REQUEST_INTERVAL = 0.3

RETRY = 3

TIMEOUT = 15

HEADERS = {
    "User-Agent":
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    " AppleWebKit/537.36 (KHTML, like Gecko)"
    " Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://finance.eastmoney.com/",
}
