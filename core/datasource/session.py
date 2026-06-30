import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from core.datasource.config import HEADERS, RETRY

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class RetrySession(requests.Session):
    def __init__(self, retries=10, backoff=1, status_forcelist=None):
        super().__init__()
        if status_forcelist is None:
            status_forcelist = [500, 502, 503, 504]
        retry_strategy = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=backoff,
            status_forcelist=status_forcelist,
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.trust_env = False  # 不使用环境变量中的代理

        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.session.headers.update(HEADERS)

    def get(self, url, params=None, timeout=15):

        r = self.session.get(url, params=params, timeout=timeout)

        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")

        return r
