from mlflow_tasks.data_handlers import path_to_dir_uri
from mlflow_tasks.data_handlers import path_to_metadata_dir_uri
from mlflow_tasks.data_handlers import path_to_exp_run_path
from mlflow_tasks.data_handlers.utility import data_handler_from_path
from mlflow_tasks import data_handlers
import mlflow

def test_path_to_exp_run_path():
    exp_id, run_id, path = path_to_exp_run_path("18/abcdefg/params/foo")
    assert (exp_id == "18") and (run_id == "abcdefg") and (path == "params/foo")

def test_data_handler_from_path():
    dh = data_handlers.Py_Obj_Handler()
    dataset = [1,2,3,5]
    run = mlflow.start_run()
    dh.register(run.info.experiment_id, run.info.run_id, "test/foo")
    dh.set(dataset)
    mlflow.end_run()
    dh.cache_local()
    # Delete object
    del dh
    # Create new handler
    dh_path = "/".join([str(x) for x in [run.info.experiment_id, run.info.run_id, "test/foo"]])
    dh2 = data_handler_from_path(dh_path)
    # Re-load the data
    dh2.register(run.info.experiment_id, run.info.run_id, "test/foo")
    assert dh2.get() == dataset
