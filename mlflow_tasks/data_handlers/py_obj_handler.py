import os
import pickle
import yaml

class Py_Obj_Handler:
    def __init__(self):
        pass

    def cache(self, result, cache_dir):
        cache_uri = os.path.join(cache_dir, "result.pkl")
        cache_file = open(cache_uri,'wb')
        pickle.dump(result, cache_file)
        cache_file.close()

        # Metadata
        metadata = {
        	"data_handler": "Py_Obj_Handler"
        }
        # Save metadata to yaml
        metadata_file_name = os.path.join(cache_dir, "meta.yaml")
        with open(metadata_file_name, 'w') as metadata_file:
        	yaml.dump(metadata, metadata_file)

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