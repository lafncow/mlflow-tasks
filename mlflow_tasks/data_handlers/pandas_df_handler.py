import os
import pandas

class Pandas_Df_Handler:
    def __init__(self):
        pass

    def cache(self, result, cache_dir):
        # TODO handle multiIndex dfs
        cache_uri = os.path.join(cache_dir, "result.csv")
        result.to_csv(cache_uri)
        return cache_dir

    def load(self, cache_dir):
        # TODO handle multiIndex dfs
        cache_uri = os.path.join(cache_dir, "result.csv")
        result = pandas.read_csv(cache_uri, index_col=0)
        return result

    def get_metrics(self, result):
        row_count = result.shape[0]
        metrics = {"result_row_count": row_count}
        return metrics