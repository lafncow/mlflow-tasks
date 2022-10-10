import os
import mlflow_tasks.data_handlers as data_handlers
from mlflow.tracking import MlflowClient

cache_dir = os.path.join(os.path.abspath(''), "mlflow_tasks_cache")


# +
def path_to_dir_uri(full_path, local_dir):
    local_array = [local_dir] + full_path.split("/") + [full_path.split("/")[-1]]
    local_uri = os.path.join(*local_array)
    
    local_array = [local_dir] + full_path.split("/")
    local_dir = os.path.join(*local_array)
    
    return (local_dir, local_uri)

def path_to_metadata_dir_uri(full_path, local_dir):
    local_dir, local_uri = path_to_dir_uri(full_path, local_dir)
    local_uri = local_uri+"_meta.yml"
    return (local_dir, local_uri)


# -

def path_to_exp_run_path(full_path):
    path_array = full_path.split('/')
    experiment_id = path_array[0]
    run_id = path_array[1]
    path = "/".join(path_array[2:])
    return (experiment_id, run_id, path)


def data_handler_from_path(full_path):
    import yaml
    mlflow_client = MlflowClient()
    experiment_id, run_id, log_path = path_to_exp_run_path(full_path)
    local_dir, local_metadata_uri = path_to_metadata_dir_uri(full_path, cache_dir)
    os.makedirs(local_dir, exist_ok=True)
    # Download metadata from log
    try:
        #print(f"DEBUG data_handler_from_path {log_path} to {local_dir} and {local_metadata_uri}")
        mlflow_client.download_artifacts(run_id, log_path, os.path.join(cache_dir, experiment_id, run_id))
    except:
        print(f"DEBUG No metadata found at {log_path}. Could not get data handler for {full_path}.")
        return None
    # Read the metadata
    with open(local_metadata_uri, 'r') as metadata_file:
        metadata = yaml.safe_load(metadata_file)
    # Find the right data handler
    data_handler_name = metadata['data_handler']
    if not data_handler_name in data_handlers.__dict__:
        raise Exception(f"Data handler {data_handler_name} not found.")
    # Create data handler
    data_handler = data_handlers.__dict__[data_handler_name](**metadata['handler_args'])
    # Load the data
    data_handler.load(metadata['full_path'])
    
    return data_handler
