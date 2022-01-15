# MLFlow Tasks

MLFlow Tasks extends MLFlow by providing a Task class that:
 1. wraps functions, files, notebooks, models, or project endpoints
 2. starts an mlflow run and common tracking
 3. logs notebooks as html artifacts
 4. handles serialization and logging of values passed in and out of runs
 5. encourages step-driven workflows
 
 For example:
 ```python
 from mlflow_tasks import Task
 
 # Run a function, arguments logged as parameters, Task object is returned
 func_task = Task(lambda x: x*2, x=42)
 print(func_task.get_result()) # -> 84
 
 # Run a script
 python_task = Task('my_script.py', some_named_input=func_task.get_result())
 
 # Run a notebook
 # HTML is logged as an artifact
 # As a shorthand, you can pass a task object
 notebook_task = Task("my_notebook.ipynb", some_input=python_task, some_other_input=42)
 
 # Task objects contain result and run information
 result = func_task.get_result() # -> 84
 experiment_id = func_task.experiment_id # -> 27
 run_id = func_task.run_id # -> 081adc5cb135404696d51f2cbd67c2f2
 status = func_task.get_run().info.status # -> 'FINISHED'
 ```

