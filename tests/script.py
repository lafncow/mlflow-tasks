from mlflow_tasks import TaskRun
from mlflow_tasks.data_handlers import PickleDataHandler

default_params = {"test_param": PickleDataHandler(21.5)}
# Fetches the active task or starts one with default params
task_run = TaskRun(experiment_name="script test",
                   handled_params=default_params)
params = task_run.get_params()
print(params)
result = params["test_param"] * 2 # 16
task_run.set_result(result)