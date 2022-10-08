from mlflow_tasks import Task

task = Task() # Fetches the active task
params = task.get_params()
print(params)
result = params["test_param"] * 2 # 16
task.set_result(result)