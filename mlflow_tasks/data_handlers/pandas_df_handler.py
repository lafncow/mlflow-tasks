import os
import pandas

class Pandas_Df_Handler:
    def __init__(self):
        pass

    def cache(self, result, cache_dir):
        # TODO handle multiIndex dfs
        cache_uri = os.path.join(cache_dir, "result.csv")
        result.to_csv(cache_uri)

        # Metadata
        metadata = {
            "data_handler": "Pandas_Df_Handler"
        }
        # Save metadata to yaml
        metadata_file_name = os.path.join(cache_dir, "meta.yaml")
        with open(metadata_file_name, 'w') as metadata_file:
            yaml.dump(metadata, metadata_file)
        
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