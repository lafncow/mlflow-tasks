import os
import pickle
from .abstract_data_handler import AbstractDataHandler


class PickleDataHandler(AbstractDataHandler):

    def __write_to_cache__(self, dataset, cache_uri):
        cache_file_uri = os.path.join(cache_uri, "dataset.pkl")
        with open(cache_file_uri, 'wb') as cache_file:
            pickle.dump(dataset, cache_file)
        return cache_uri

    def __read_from_cache__(self, cache_uri):
        dataset = None
        cache_file_uri = os.path.join(cache_uri, "dataset.pkl")
        print(cache_file_uri)
        with open(cache_file_uri, 'rb') as cache_file:
            dataset = pickle.load(cache_file)
        return dataset