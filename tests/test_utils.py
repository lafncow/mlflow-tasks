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
import mlflow_tasks

# +
# Test Utility Functions
def test_util_create_experiment():
    experiment = mlflow_tasks.get_or_create_experiment("Test Experiment")
    assert experiment.name == "Test Experiment"

def test_util_get_experiment():
    experiment = mlflow_tasks.get_or_create_experiment("Test Experiment")
    assert experiment.name == "Test Experiment"

def test_active_task_none():
    task = mlflow_tasks.active_task()
    assert task is None

def test_start_task():
    task = mlflow_tasks.start_task()
    assert isinstance(task, mlflow_tasks.Task)
    
def test_active_task_exists():
    mlflow_tasks.start_task()
    task = mlflow_tasks.active_task()
    assert isinstance(task, mlflow_tasks.Task)

def test_get_task():
    mlflow_tasks.start_task()
    act_task = mlflow_tasks.active_task()
    task = mlflow_tasks.get_task(act_task.run_id)
    task.end_run()
    assert isinstance(task, mlflow_tasks.Task)
