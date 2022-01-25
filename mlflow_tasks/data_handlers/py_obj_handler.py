import os
import pickle

class Py_Obj_Handler:
    def __init__(self):
        pass

    def cache(self, result, cache_dir):
        cache_uri = os.path.join(cache_dir, "result.pkl")
        cache_file = open(cache_uri,'wb')
        pickle.dump(result, cache_file)
        cache_file.close()
        return cache_dir

    def load(self, cache_dir):
        cache_uri = os.path.join(cache_dir, "result.pkl")
        cache_file = open(cache_uri, 'rb')
        result = pickle.load(cache_file)
        cache_file.close()
        return result

    def get_metrics(self, result):
        metrics = {}
        return metrics