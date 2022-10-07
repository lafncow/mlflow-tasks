import os
import pickle
import yaml
import mlflow
from mlflow.tracking import MlflowClient

class Py_Obj_Handler:
    def __init__(self, cache_dir=None, experiment_id=None, run_id=None, path=None):
        self.__data__ = None
        self.full_path = None
        self.path = path
        self.log_uri = None
        self.local_cache_uri = None
        self.global_cache_uri = None
        self.experiment_id = experiment_id
        self.run_id = run_id
        
        if (not self.experiment_id is None) and (not self.run_id is None) and (not self.path is None):
            self.full_path = "/".join([self.experiment_id, self.run_id, self.path])
        
        if cache_dir is None:
            cache_dir = os.path.join(os.path.abspath(''), "mlflow_tasks_cache")
        self.cache_dir = cache_dir
        
        self.mlflow_client = MlflowClient()

    def load(self, full_path):
        if full_path is None:
            return None
        self.full_path = full_path

        # Check local cache
        # Create cache uri
        local_cache_uri = [self.cache_dir] + self.full_path.split("/")
        local_cache_uri = os.path.join(*local_cache_uri)
        # Pull data from cache
        try:
            cache_file = open(local_cache_uri, 'rb')
            dataset = pickle.load(cache_file)
            self.__data__ = dataset
            cache_file.close()
            # Set cache uri
            self.local_cache_uri = local_cache_uri
        except:
            pass

        if self.__data__ is None:
            # Check global cache
            # Check log
            # Create log uri
            
            log_path = full_path.split("/")[2:]
            # Download from log
            self.mlflow_client.download_artifact(self.run_id, log_path, local_cache_uri)
            cache_file = open(local_cache_uri, 'rb')
            dataset = pickle.load(cache_file)
            self.__data__ = dataset
            cache_file.close()
            # Set cache uri
            self.local_cache_uri = local_cache_uri
            self.global_cache_uri = log_path
            self.log_uri = log_path
        
        self.full_path = full_path
        self.experiment_id = full_path.split("/")[0]
        self.run_id = full_path.split("/")[1]
        self.path = full_path.split("/")[2:]
        
        return self.__data__
    
    def cache_local(self):
        # Write to cache accessible inside of this machine
        local_cache_array = [self.cache_dir] + self.full_path.split("/")
        local_cache_uri = os.path.join(*local_cache_array)
        local_cache_dir = os.path.split(local_cache_uri)[:-1]
        local_cache_dir = os.path.join(*local_cache_dir)
        os.makedirs(local_cache_dir, exist_ok=True)
        cache_file = open(local_cache_uri,'wb')
        pickle.dump(self.__data__, cache_file)
        cache_file.close()
        
        # Create Metadata
        metadata = {
            "data_handler": "Py_Obj_Handler",
            "handler_args": {
                "cache_dir": self.cache_dir,
                "experiment_id": self.experiment_id,
                "run_id": self.run_id,
                "path": self.path
            },
            "full_path": self.full_path
        }
        # Save metadata to yaml
        metadata_file_name = self.full_path.split("/")[-1] + "_meta.yaml"
        metadata_path = [self.cache_dir] + self.full_path.split("/")[:-1] + [metadata_file_name]
        metadata_path = os.path.join(*metadata_path)
        with open(metadata_path, 'w') as metadata_file:
            yaml.dump(metadata, metadata_file)
        self.mlflow_client.log_artifact(self.run_id, metadata_path)
        
        # Set cache uri
        self.local_cache_uri = local_cache_uri
        
        return self.local_cache_uri

    def log(self):
        # Save to local dir first
        if self.local_cache_uri is None:
            self.cache_local()
             
        log_uri = "/".join(self.full_path.split("/")[2:])
        self.mlflow_client.log_artifact(self.run_id, self.local_cache_uri, log_uri)
        
        self.log_uri = log_uri

        return self.log_uri
    
    def cache_global(self):
        # Write to cache accessible outside of this machine
        log_uri = self.log()
        return log_uri

    def set(self, dataset, experiment_id, run_id, path):
        self.__data__ = dataset
        self.path = path
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.full_path = "/".join([str(x) for x in [experiment_id, run_id, path]])

        return self
    
    def get(self):
        if self.__data__ is None:
            self.load(self.full_path)
        return self.__data__