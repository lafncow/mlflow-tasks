# +
import os
import sys
import tempfile

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


# +
# Test Task Methods
def test_task_init():
    task = mlflow_tasks.Task(experiment_name="test_task_init")
    assert task.experiment.name == "test_task_init"


def test_task_run():
    task = mlflow_tasks.Task(experiment_name="test_task_run")
    task_run = task.run()
    assert isinstance(task_run, mlflow_tasks.TaskRun)


def test_taskrun_init_unnamed():
    task_run = mlflow_tasks.TaskRun()
    assert isinstance(task_run, mlflow_tasks.TaskRun)


def test_taskrun_init_named():
    task_run = mlflow_tasks.TaskRun(experiment_name="test_taskrun_init_named")
    experiment_id = task_run.mlflow_run.info.experiment_id
    experiment = mlflow.get_experiment(experiment_id)
    assert experiment.name == "test_taskrun_init_named"


def test_task_run_multiple():
    task = mlflow_tasks.Task(experiment_name="test_task_run")
    task_runs = [task.run(x=i) for i in range(3)]
    assert isinstance(task_runs[0], mlflow_tasks.TaskRun)


def test_taskrun_get_params():
    task = mlflow_tasks.Task(experiment_name="test_taskrun_get_params")
    task_run = task.run(x=1, y=2)
    params = task_run.get_params()
    assert params == {'x': 1, 'y': 2}


def test_taskrun_set_result():
    task = mlflow_tasks.Task(experiment_name="test_taskrun_set_result")
    task_run = task.run()
    task_run.set_result(8)
    assert task_run.get_result() == 8


def test_taskrun_none_result():
    task = mlflow_tasks.Task(experiment_name="test_taskrun_none_result")
    task_run = task.run()
    assert task_run.get_result() is None


# ========================================================================


def test_func_task():
    task = mlflow_tasks.FuncTask(lambda x: x-1,
                                 experiment_name="test_func_task")
    task_run = task.run(x=8)
    res = task_run.get_result()
    assert res == 7


def test_py_task():
    task = mlflow_tasks.PyTask("tests/script.py",
                               experiment_name="test_task_exec_script")
    task_run = task.run(test_param=8)
    res = task_run.get_result()
    assert res == 16


# Define the model class
class AddN(mlflow.pyfunc.PythonModel):

    def __init__(self, n):
        self.n = n

    def predict(self, context, model_input):
        return model_input.apply(lambda column: column + self.n)


def test_model_task():
    # Define model
    my_model = AddN(n=5)
    # Save model
    temp_path = tempfile.TemporaryDirectory().name
    model_path = os.path.join(temp_path, "pytest_test_model")
    mlflow.pyfunc.save_model(model_path,
                             python_model=my_model)
    # Create input
    import pandas as pd
    model_input = pd.DataFrame([range(10)])
    # Create task
    task = mlflow_tasks.ModelTask(model_path,
                                  experiment_name="test_model_task")
    task_run = task.run(model_input=model_input)
    res = task_run.get_result()
    assert res.equals(pd.DataFrame([range(10)]) + 5)


def test_taskrun_input_to_model():
    # Define model
    my_model = AddN(n=5)
    # Save model
    temp_path = tempfile.TemporaryDirectory().name
    model_path = os.path.join(temp_path, "pytest_test_model")
    mlflow.pyfunc.save_model(model_path,
                             python_model=my_model)
    # Create input
    import pandas as pd

    def get_df():
        return pd.DataFrame([range(10)])

    input_task = mlflow_tasks.FuncTask(get_df,
                                       experiment_name="test_task_input_to_model_1")
    input_task_run = input_task.run()
    # Create task
    model_task = mlflow_tasks.ModelTask(model_path,
                                   experiment_name="test_task_input_to_model_2")
    task_run = model_task.run(model_input=input_task_run)
    res = task_run.get_result()
    assert res.equals(pd.DataFrame([range(10)]) + 5)


#def test_nb_task():
#    task = mlflow_tasks.NotebookTask("tests/notebook.ipynb",
#                                     experiment_name="test_task_exec_nb")
#    task_run = task.run(test_param=8)
#    res = task_run.get_result()
#    assert res == 16


def test_project_task_path_only():
    task = mlflow_tasks.ProjectTask("tests/project",
                                    experiment_name="test_project_task_path_only")
    task_run = task.run(test_param=8)
    res = task_run.get_result()
    assert res == 16