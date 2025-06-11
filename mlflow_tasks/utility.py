import tempfile
import os
import mlflow
from mlflow import MlflowClient
import mlflow_tasks.data_handlers as data_handlers


def get_or_create_experiment(experiment_name: str):
    """
    Retrieve an existing MLflow experiment or create it if it does not exist.

    Checks MLflow for an experiment with the specified name. If it does not exist,
    a new experiment is created and its details are returned.

    Args:
        experiment_name (str): Name of the experiment to retrieve or create.

    Returns:
        mlflow.entities.Experiment: The MLflow experiment object corresponding to
        the given name, either retrieved or newly created.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment {experiment_name} does not exist on MLFlow yet, creating it...")
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment(experiment_id)

    return experiment


def data_handler_from_path(log_path: str):
    """
    Retrieve a DataHandler instance from an MLflow artifact path.

    Parses an MLflow artifact path to dynamically instantiate and configure a
    DataHandler using metadata stored in the experiment tracking system. Designed
    for loading persisted data handlers from previous task runs.

    Args:
        log_path (str): MLflow artifact path in "run_id/data_name" format.
            The location where the data handler's metadata YAML is stored.

    Returns:
        DataHandler: Configured data handler instance ready for use, or None
        if the metadata artifact cannot be found in the MLFlow run log.

    Raises:
        Exception: If the data handler class specified in metadata cannot be
        found in the registered data handlers.

    Example:
        >>> handler = data_handler_from_path("abc12345678675309/my_dataset")
        >>> my_dataset = handler.get()

    Note:
        Requires MLflow server connectivity to fetch artifacts. Creates temporary
        directories that are automatically cleaned up.
    """
    # log_path is in the format: "run_id/data_name"
    import yaml
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name
    mlflow_client = MlflowClient()
    run_id, data_name = (log_path.split("/")[0], "/".join(log_path.split("/")[1:]))
    # Check if the metadata exists before downloading
    # List artifacts
    artifacts_list = mlflow_client.list_artifacts(run_id)
    found_metadata = False
    # Check if the metadata file exists
    for artifact in artifacts_list:
        print(f"Checking artifact {artifact.path}")
        if artifact.path == data_name.split("/")[0]:
            found_metadata = True
            print("Found!")
            break
    if found_metadata is False:
        # If it does not exist, return None
        return None

    # Download metadata from log
    metadata_file_path = mlflow_client.download_artifacts(run_id, f"{data_name}/data_handler.yml", temp_dir)
    # Read the metadata
    with open(metadata_file_path, 'r') as metadata_file:
        metadata = yaml.safe_load(metadata_file)
    # Find the right data handler
    data_handler_name = metadata['data_handler']
    if data_handler_name not in data_handlers.__dict__:
        raise Exception(f"Data handler {data_handler_name} not found.")
    # Create data handler
    handler_args = {
        "data_name": metadata["data_name"],
        "load_from": metadata["data_source"],
        "cache_dir": metadata["cache_dir"],
        "use_cache": metadata["use_cache"],
        "use_log": metadata["use_log"]
    }
    data_handler = data_handlers.__dict__[data_handler_name](**handler_args)

    # Configure the data handler
    data_handler.register(metadata["run_id"])
    data_handler.log_path = metadata["log_path"]
    data_handler.cache_uri = metadata["cache_uri"]

    return data_handler