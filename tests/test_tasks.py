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

import mlflow_tasks
import mlflow

test_tracking_folder = os.path.join(project_folder, "test_mlruns")
# Clear the testing tracking folder
import shutil
try:
    shutil.rmtree(test_tracking_folder)
except:
    pass

# Set the mlflow tracking folder
mlflow.set_tracking_uri(f"file:/{test_tracking_folder}")


# +
# Test Task Methods
def test_create_task():
    task = mlflow_tasks.Task(experiment_name="Task Test Experiment")
    task.end_run()
    assert isinstance(task, mlflow_tasks.Task)

def test_create_task_named():
    task = mlflow_tasks.Task(experiment_name="Task Test Experiment")
    task.end_run()
    assert task.experiment_name == "Task Test Experiment"

def test_task_get_run():
    task = mlflow_tasks.Task(experiment_name="Task Test Experiment")
    run = task.get_run()
    task.end_run()
    assert isinstance(run, mlflow.entities.Run)
    
def test_task_end_task():
    task = mlflow_tasks.Task(experiment_name="Task Test Experiment")
    task.end_run()
    assert task.get_run().info.status == "FINISHED"

def test_task_set_result():
    task = mlflow_tasks.Task(experiment_name="Task Test Experiment")
    task.set_result(8)
    task.end_run()
    assert task.result == 8

def test_task_get_result():
    task = mlflow_tasks.Task(experiment_name="Task Test Experiment")
    task.set_result(8)
    task.end_run()
    res = task.get_result()
    assert res == 8

def test_task_get_result_from_cache():
    task = mlflow_tasks.Task(experiment_name="Task Test Experiment")
    task.set_result(8)
    task.end_run()
    # Clear result
    task.result = None
    res = task.get_result()
    assert res == 8

def test_task_get_params():
    task = mlflow_tasks.Task(lambda x:x-1, x=8, experiment_name="Task Test Experiment")
    params = task.get_params()
    task.end_run()
    assert params['x'] == 8

def test_task_exec_func():
    task = mlflow_tasks.Task(lambda x:x-1, x=8, experiment_name="Task Test Experiment")
    res = task.get_result()
    task.end_run()
    assert res == 7