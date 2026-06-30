import time
from core.datasource.config import REQUEST_INTERVAL


class RateLimiter:

    def __init__(self):

        self.last = 0

    def wait(self):

        now = time.time()

        diff = now - self.last

        if diff < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - diff)

        self.last = time.time()
