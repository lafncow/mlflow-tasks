# +
import os
import sys

# Add the folder that contains this script to PYTHONPATH so that mlflow_tasks can be imported
try:
    #__file__
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except:
    project_folder = os.path.split(os.path.abspath(''))[0]

sys.path.insert(0, project_folder)

import mlflow
from mlflow.tracking import MlflowClient

test_tracking_folder = os.path.join(project_folder, "test_mlruns")
# Clear the testing tracking folder
import shutil
try:
    #shutil.rmtree(test_tracking_folder)
    os.mkdir(test_tracking_folder)
except:
    pass
# Make sure testing tracking folder exists
try:
    os.mkdir(test_tracking_folder)
except:
    pass
test_tracking_folder = os.path.join(project_folder, "test_mlruns")

# Set the mlflow tracking folder
mlflow.set_tracking_uri(f"file:/{test_tracking_folder}")

mlflow_client = MlflowClient()
from mlflow_tasks import data_handlers

# +

def test_data_handler_set():
    dh = data_handlers.Py_Obj_Handler()
    dataset = [1,2,3,5]
    run = mlflow.start_run()
    dh.set(dataset, run.info.experiment_id, run.info.run_id, "test/foo")
    mlflow.end_run()
    assert dh.__data__ == dataset

def test_data_handler_get():
    dh = data_handlers.Py_Obj_Handler()
    dataset = [1,2,3,5]
    run = mlflow.start_run()
    dh.set(dataset, run.info.experiment_id, run.info.run_id, "test/foo")
    mlflow.end_run()
    assert dh.get() == dataset

def test_data_handler_cache_local():
    dh = data_handlers.Py_Obj_Handler()
    dataset = [1,2,3,5]
    run = mlflow.start_run()
    dh.set(dataset, run.info.experiment_id, run.info.run_id, "test/foo")
    mlflow.end_run()
    dh.cache_local()
    # Delete object
    del dh
    # Create new handler
    dh2 = data_handlers.Py_Obj_Handler()
    # Re-load the data
    dh2.load("/".join([run.info.experiment_id, run.info.run_id,"test/foo"]))
    assert dh2.get() == dataset

def test_data_handler_cache_global():
    dh = data_handlers.Py_Obj_Handler()
    dataset = [1,2,3,5]
    run = mlflow.start_run()
    dh.set(dataset, run.info.experiment_id, run.info.run_id, "test/foo")
    mlflow.end_run()
    dh.cache_global()
    # Delete object
    del dh
    # Create new handler
    dh2 = data_handlers.Py_Obj_Handler()
    # Re-load the data
    dh2.load("/".join([run.info.experiment_id, run.info.run_id,"test/foo"]))
    assert dh2.get() == dataset

def test_data_handler_log():
    dh = data_handlers.Py_Obj_Handler()
    dataset = [1,2,3,5]
    run = mlflow.start_run()
    dh.set(dataset, run.info.experiment_id, run.info.run_id, "test/foo")
    mlflow.end_run()
    dh.log()
    # Delete object
    del dh
    # Create new handler
    dh2 = data_handlers.Py_Obj_Handler()
    # Re-load the data
    dh2.load("/".join([run.info.experiment_id, run.info.run_id,"test/foo"]))
    assert dh2.get() == dataset
