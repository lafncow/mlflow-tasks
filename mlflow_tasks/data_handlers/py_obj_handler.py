import os
import pickle
import yaml
import mlflow
from .utility import *
from mlflow.tracking import MlflowClient

class Py_Obj_Handler:
    def __init__(self, cache_dir=None):
        self.__data__ = None
        self.full_path = None
        self.path = None
        self.log_uri = None
        self.local_cache_uri = None
        self.global_cache_uri = None
        self.experiment_id = None
        self.run_id = None
        if cache_dir is None:
            cache_dir = os.path.join(os.path.abspath(''), "mlflow_tasks_cache")
        self.cache_dir = cache_dir
        
        self.mlflow_client = MlflowClient()

    def register(self, experiment_id, run_id, path):
        # Sets the experiment, run, and relative path
        self.full_path = "/".join([str(x) for x in [experiment_id, run_id, path]])
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.path = path
    
    def cache_local(self):
        # TODO check if already cached
        # Write to cache accessible inside of this machine
        local_cache_dir, local_cache_uri = path_to_dir_uri(self.full_path, self.cache_dir)
        
        os.makedirs(local_cache_dir, exist_ok=True)
        with open(local_cache_uri,'wb') as cache_file:
            pickle.dump(self.__data__, cache_file)
        
        # Create Metadata
        metadata = {
            "data_handler": "Py_Obj_Handler",
            "handler_args": {
                "cache_dir": self.cache_dir
            },
            "full_path": self.full_path,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "path": self.path
        }
        
        # Save metadata to yaml
        local_meta_dir, local_meta_uri = path_to_metadata_dir_uri(self.full_path, self.cache_dir)
        
        with open(local_meta_uri, 'w') as metadata_file:
            yaml.dump(metadata, metadata_file)
        self.mlflow_client.log_artifact(self.run_id, local_meta_uri, self.path)
        #print(f"DEBUG DH.cache_local {self.path} ({self.__data__}) to {local_cache_uri} and {local_meta_uri}")
        # Set cache uri
        self.local_cache_uri = local_cache_uri
        
        return self.local_cache_uri

    def log(self):
        # TODO check if already logged
        # Save to local dir first
        if self.local_cache_uri is None:
            self.cache_local()
        
        self.mlflow_client.log_artifact(self.run_id, self.local_cache_uri, self.path)
        self.log_uri = self.full_path

        return self.full_path
    
    def cache_global(self):
        # Write to cache accessible outside of this machine
        log_uri = self.log()
        self.global_cache_uri = log_uri
        return log_uri

    def set(self, dataset):
        self.__data__ = dataset
    
    def get(self):
        if not self.__data__ is None:
            return self.__data__
        # Check local cache
        # Create cache uri
        local_cache_dir, local_cache_uri = path_to_dir_uri(self.full_path, self.cache_dir)
        # Pull data from cache
        try:
            #print(f"DEBUG DH.load cache seek: {local_cache_uri}")
            with open(local_cache_uri, 'rb') as cache_file:
                self.__data__ = pickle.load(cache_file)
            # Set cache uri
            self.local_cache_uri = local_cache_uri
        except:
            #print(f"DEBUG DH.load cache miss: {local_cache_uri}")

            try:
                # Check global cache
                # Check log
                # Create log uri

                # Download from log

                #print(f"DEBUG DH.load download {self.path} to {local_cache_dir}")
                os.makedirs(local_cache_dir, exist_ok=True)
                self.mlflow_client.download_artifacts(self.run_id, self.path, os.path.join(self.cache_dir, self.experiment_id, self.run_id))

                with open(local_cache_uri, 'rb') as cache_file:
                    self.__data__ = pickle.load(cache_file)

                # Set cache uri
                self.local_cache_uri = local_cache_uri
                self.global_cache_uri = self.path
                self.log_uri = self.path
            except:
                pass
        
        return self.__data__