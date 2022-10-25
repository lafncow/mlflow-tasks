import os
import sys

try:
    #__file__
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
except:
    project_folder = os.path.split(os.path.abspath(''))[0]

sys.path.insert(0, project_folder)

from mlflow_tasks import Task

task = Task() # Fetches the active task
params = task.get_params()
print(params)
result = params["test_param"] * 2 # 16
task.set_result(result)