import os
import sys

try:
    #__file__
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
except:
    project_folder = os.path.split(os.path.abspath(''))[0]

sys.path.insert(0, project_folder)

from mlflow_tasks import TaskRun
from mlflow_tasks.data_handlers import PickleDataHandler

default_params = {"test_param": PickleDataHandler(21.5)}
task_run = TaskRun(experiment_name="project test",
                   handled_params=default_params) # Fetches the active task
params = task_run.get_params()
result = params["test_param"] * 2 # 16
task_run.set_result(result)