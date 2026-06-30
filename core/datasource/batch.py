from concurrent.futures import ThreadPoolExecutor


class BatchDownloader:

    def __init__(self, datasource, workers=8):

        self.ds = datasource

        self.workers = workers

    def download_kline(self, codes, start, end):

        results = {}

        def job(code):

            return code, self.ds.stock_hist(code, start, end)

        with ThreadPoolExecutor(self.workers) as ex:

            for code, df in ex.map(job, codes):

                results[code] = df

        return results
