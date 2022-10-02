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

from mlflow.tracking import MlflowClient
mlflow_client = MlflowClient()
import mlflow_tasks
from mlflow_tasks import data_handlers

# +

def test_task_pass_data_to_func():
    def read_func(x):
        assert x == [1,2,3]
    task = mlflow_tasks.Task(read_func, x=[1,2,3])

def test_task_param_to_func_get_params():
    t = mlflow_tasks.Task()

    t.set_result([1,2,3])
    t.end_run()

    def x(foo):
        print(foo)
    t2 = mlflow_tasks.Task(x, foo=t)
    t2_params = t2.get_params()
    assert {'foo':[1,2,3]} == t2_params

def test_task_local_cache_get_result():
    t = mlflow_tasks.Task(write_local_cache=True)

    t.set_result([1,2,3])
    run_id = t.run_id
    t.end_run()
    del t

    t2 = mlflow_tasks.Task(run_id = run_id)
    t2.end_run()
    assert [1,2,3] == t2.get_result()

def test_task_global_cache_get_result():
    t = mlflow_tasks.Task(write_global_cache=True)

    t.set_result([1,2,3])
    run_id = t.run_id
    t.end_run()
    del t

    t2 = mlflow_tasks.Task(run_id = run_id)
    t2.end_run()
    assert [1,2,3] == t2.get_result()
    
def test_task_log_get_result():
    t = mlflow_tasks.Task(write_log=True)

    t.set_result([1,2,3])
    run_id = t.run_id
    t.end_run()
    del t

    t2 = mlflow_tasks.Task(run_id = run_id)
    t2.end_run()
    assert [1,2,3] == t2.get_result()

def test_pass_cached_reloaded_task():
    t = mlflow_tasks.Task(write_local_cache=True)

    t.set_result([1,2,3])
    run_id = t.run_id
    t.end_run()
    del t

    t = mlflow_tasks.Task(run_id = run_id)
    t.end_run()
    def x(foo):
        return foo
    t2 = mlflow_tasks.Task(x, foo=t)
    t2.end_run()
    assert [1,2,3] == t2.get_result()