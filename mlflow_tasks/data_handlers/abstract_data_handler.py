import os
import yaml
from mlflow.tracking import MlflowClient
import tempfile
from abc import ABC, abstractmethod


class AbstractDataHandler(ABC):
    """
    Abstract base class for handling data persistence and retrieval with MLflow integration.

    This class provides a framework for managing datasets in memory, caching them to disk,
    logging them to MLflow experiments, and reloading them from various sources. It supports
    flexible configuration for caching and logging behavior, and integrates with MLflow for
    experiment tracking and artifact storage.

    Attributes:
        mlflow_run (mlflow.entities.Run): The associated MLflow run object.
        data_name (str): Name or path identifier for the dataset.
        data_source (str): Source URI for loading the dataset.
        cache_dir (str): Directory path for caching the dataset.
        temp_dir (str): Temporary directory for transient storage.
        use_cache (bool): Whether to use disk caching.
        use_log (bool): Whether to log data to MLflow.
        cache_uri (str): Full path to the cache location.
        log_path (str): Path for artifact logging in MLflow.

    Methods:
        register(run_id): Associate the handler with an MLflow run.
        cache(): Cache the dataset to disk and log metadata.
        temp_cache(): Cache the dataset to a temporary directory and log metadata.
        log(): Log the dataset and its metadata to MLflow.
        set(dataset): Store the dataset in memory and optionally cache/log it.
        reload_metadata(): Reload handler metadata from MLflow artifacts.
        get(): Retrieve the dataset from memory, cache, log, or source.
        __write_to_cache__(dataset, cache_path): Abstract method to write dataset to cache.
        __read_from_cache__(cache_path): Abstract method to read dataset from cache.

    Abstract Methods:
        __write_to_cache__(dataset, cache_path): Must be implemented to define how data
            is written to cache.
        __read_from_cache__(cache_path): Must be implemented to define how data is read
            from cache.
    """
    __data_set__ = None

    mlflow_run = None
    data_name = None
    data_source = None
    cache_dir = None
    temp_dir = None
    use_cache = False
    use_log = False

    def __init__(self, dataset=None, data_name=None,
                 load_from=None, cache_dir=None,
                 use_cache=False, use_log=False):
        # TODO Clean this up
        self.__data_set__ = dataset
        self.__mlflow_client__ = MlflowClient()
        temp_dir_obj = tempfile.TemporaryDirectory()
        self.__temp_dir__ = temp_dir_obj.name
        # namespace and name of the data, ex: 'params/my_data'
        if data_name is None:
            # Create random name
            import uuid
            self.data_name = uuid.uuid4().hex
        else:
            self.data_name = data_name
        # uri for loading the data, to be fed to __read_from_cache__
        self.data_source = load_from
        # full path for a cache location, used as indicator that data is cached
        self.cache_uri = None
        # registered mlflow_run
        self.mlflow_run = None
        # path to cache directory
        if cache_dir is not None:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = self.__temp_dir__
        # full path for a temp location, used as indicator that data is temp cached
        self.__temp_uri__ = None
        self.log_path = None
        self.use_cache = use_cache
        self.use_log = use_log

    def register(self, run_id):
        # Sets the data name and data path
        self.mlflow_run = self.__mlflow_client__.get_run(run_id)

    @abstractmethod
    def __write_to_cache__(self, dataset, cache_path):
        pass

    @abstractmethod
    def __read_from_cache__(self, cache_path):
        pass

    def cache(self):
        print(f"Caching to {self.cache_dir}")
        # Check if we have a registered run
        if self.mlflow_run is None:
            raise Exception("The Data Handler must be associated to an MLFlow Run (use .set_run(RUN_ID))")

        run_id = self.mlflow_run.info.run_id
        experiment_id = self.mlflow_run.info.experiment_id
        corrected_data_name = os.path.join(*self.data_name.split("/"))
        self.cache_uri = os.path.join(self.cache_dir,
                                      str(experiment_id),
                                      str(run_id),
                                      corrected_data_name)
        os.makedirs(self.cache_uri, exist_ok=True)
        # Write to cache
        self.__write_to_cache__(self.__data_set__, self.cache_uri)
        self.__log_metadata__()

        return self.cache_uri

    def temp_cache(self):
        print(f"Caching {self.data_name} to {self.__temp_dir__}")
        # Check if we have a registered run
        if self.mlflow_run is None:
            raise Exception("The Data Handler must be associated to an MLFlow Run (use .set_run(RUN_ID))")

        run_id = self.mlflow_run.info.run_id
        experiment_id = self.mlflow_run.info.experiment_id
        corrected_data_name = os.path.join(*self.data_name.split("/"))
        self.__temp_uri__ = os.path.join(self.__temp_dir__,
                                         str(experiment_id),
                                         str(run_id),
                                         corrected_data_name)
        os.makedirs(self.__temp_uri__, exist_ok=True)
        self.cache_uri = self.__temp_uri__
        # Write to local temp
        self.__write_to_cache__(self.__data_set__, self.__temp_uri__)
        self.__log_metadata__()

        return self.__temp_uri__

    def __log_metadata__(self):
        # Check if we have a associated run
        if self.mlflow_run is None:
            raise Exception("The Data Handler must be associated to an MLFlow Run (use .set_run(RUN_ID))")
        # Prepare metadata
        metadata = {
            "data_handler": type(self).__name__,
            "experiment_id": self.mlflow_run.info.experiment_id,
            "run_id": self.mlflow_run.info.run_id,
            "data_name": self.data_name,
            "data_source": self.data_source,
            "cache_dir": self.cache_dir,
            "cache_uri": self.cache_uri,
            "log_path": self.log_path,
            "use_cache": self.use_cache,
            "use_log": self.use_log
        }

        # Save metadata to yaml in temp
        run_id = self.mlflow_run.info.run_id
        experiment_id = self.mlflow_run.info.experiment_id
        corrected_data_name = os.path.join(*self.data_name.split("/"))
        meta_file_name = "data_handler.yml"
        metadata_uri = os.path.join(self.__temp_dir__,
                                    run_id,
                                    experiment_id,
                                    corrected_data_name,
                                    meta_file_name)
        os.makedirs(os.path.split(metadata_uri)[0], exist_ok=True)

        with open(metadata_uri, 'w') as metadata_file:
            yaml.dump(metadata, metadata_file)

        # Log metadata
        self.__mlflow_client__.log_artifact(self.mlflow_run.info.run_id, metadata_uri, self.data_name)
        print(f"DEBUG logging {self.data_name} metadata (data_handler.yml) to {self.mlflow_run.info.run_id}")

    def log(self):
        # Check if we have a registered run
        if self.mlflow_run is None:
            raise Exception("The Data Handler must be associated to an MLFlow Run (use .set_run(RUN_ID))")

        # Save to local dir first
        if self.__temp_uri__ is None:
            self.temp_cache()
        print(f"Logging to {self.data_name}")
        self.__mlflow_client__.log_artifacts(self.mlflow_run.info.run_id, self.__temp_uri__, self.data_name)

        self.log_path = self.data_name
        self.__log_metadata__()

        return self.data_name

    def set(self, dataset):
        # Save dataset to memory
        self.__data_set__ = dataset

        if self.use_cache:
            self.cache()
        if self.use_log:
            self.log()

    def reload_metadata(self):
        run_id = self.mlflow_run.info.run_id
        # Check if the metadata is logged
        artifacts_list = self.__mlflow_client__.list_artifacts(run_id)
        found_metadata = False
        for artifact in artifacts_list:
            print(f"Checking artifact {artifact.path}")
            if artifact.path == self.data_name:
                found_metadata = True
                print("Found!")
                break
        if found_metadata == False:
            return None
        # Download metadata from log
        self.__mlflow_client__.download_artifacts(run_id, f"{self.data_name}/data_handler.yml", self.__temp_dir__)
        metadata_file_path = os.path.join(self.__temp_dir__, self.data_name, "data_handler.yml")
        # Read the metadata
        with open(metadata_file_path, 'r') as metadata_file:
            metadata = yaml.safe_load(metadata_file)
        self.log_path = metadata["log_path"]
        self.data_source = metadata["data_source"]
        self.cache_dir = metadata["cache_dir"]
        self.cache_uri = metadata["cache_uri"]
        self.use_cache = metadata["use_cache"]
        self.use_log = metadata["use_log"]

    def get(self):
        # Pull data from memory
        if self.__data_set__ is not None:
            # Already have the data
            return self.__data_set__

        # Pull data from temp
        if self.__temp_uri__ is not None:
            self.__data_set__ = self.__read_from_cache__(self.__temp_uri__)
            return self.__data_set__

        # Refresh metadata from log
        self.reload_metadata()

        # Pull data from cache
        if self.cache_uri is not None:
            # Pull data from cache
            self.__data_set__ = self.__read_from_cache__(self.cache_uri)

        # Pull data from log
        elif self.log_path is not None:
            # Download from log to temp
            temp_uri = os.path.join(self.__temp_dir__,
                                    str(self.mlflow_run.info.experiment_id),
                                    str(self.mlflow_run.info.run_id))
            os.makedirs(temp_uri, exist_ok=True)
            self.__temp_uri__ = self.__mlflow_client__.download_artifacts(self.mlflow_run.info.run_id,
                                                  self.log_path, temp_uri)

            # Pull data from temp
            self.__data_set__ = self.__read_from_cache__(self.__temp_uri__)

        # Pull data from source
        elif self.data_source is not None:
            # Pull data from cache
            self.__data_set__ = self.__read_from_cache__(self.data_source)

        if self.__data_set__ is None:
            print("DEBUG Data not found in memory, temp, cache, source, or log!")

        return self.__data_set__