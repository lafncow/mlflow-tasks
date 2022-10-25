# MLFlow Tasks

MLFlow Tasks extends [MLFlow](https://github.com/mlflow/mlflow) to create task-driven workflows. It lets you break up code into small tasks, while tracking everything with MLFlow.

**This project is in Alpha status, APIs may change without warning. Please use Github Issues to suggest improvements.**

MFlow Tasks provides two classes, **Task** and **Flow**:

## Task

A class to run workflow tasks by:
 1. executing functions, files, notebooks, MLFlow projects, or MLFLow models as tasks
 2. passing data in and out of tasks, using caching and mlflow logging as needed
 3. logging the task execution as an MLFlow run
 4. logging helpful parameters, metrics, and artifacts (for example: logging notebook tasks as html)

Task can be used inside orchestration scripts to run workflow tasks:
```python
from mlflow_tasks import Task

task = Task("my_task.py", my_param=8)
task_result = task.get_result() # 16
```
...and inside task scripts to access the same Task object:
```python
# my_task.py
from mlflow_tasks import Task

task = Task() # Fetches the active task
params = task.get_params()
result = param["my_param"] * 2 # 16
task.set_result(result)
```

## Flow

A subclass of Task; provides a "main" task that tracks all of the sub tasks in a workflow.

```python
from mlflow_tasks import Flow

flow = Flow(experiment_name="My Workflow")
task1 = flow.start_task("task1.py")
task2 = flow.start_task("task2.py")
task3 = flow.start_task("task3.py")
flow.end_flow()
```

## Installation

For now, MlFlow Tasks must be installed from source, using setup.py:
 ```python
pip install .
 ```

## Task Types
Tasks can be Python functions, .py scripts, .ipynb notebooks, MLFlow projects, or MLFlow models

### Python Function
```python
func_task = Task(lambda x: x*2, x=8)
```

### Python Script
```python
py_task = Task("my_task.py", x=8)
```

### iPython Notebook
```python
nb_task = Task("my_task.ipynb", x=8)
```

### MLFLow Project
```python
task = Task(("path/to/project", "entry_point"), x=8) # Tuple of path and entry_point
```

### MLFlow Model
```python
mlflow.pyfunc.log_model("my_model", python_model=my_model)
model_uri = mlflow.get_artifact_uri("my_model")
model_task = Task(model_uri, model_input=pd.DataFrame([range(10)])) # input must match MLFlow model input schema
```
...or for registered models:
```python
model_task = Task("models:/my_model/1", model_input=pd.DataFrame([range(10)])) # input must match MLFlow model input schema
```