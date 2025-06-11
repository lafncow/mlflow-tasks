from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import os
import mlflow

from . import data_handlers
from .data_handlers.abstract_data_handler import AbstractDataHandler
from .utility import get_or_create_experiment
from .utility import data_handler_from_path


ACTIVE_TASK_STACK = []


class TaskRun:
    """
    Class for Runs: instances of Tasks that are executing and logged as Runs in MLFlow
    Goals:
     - Represent a running unit of work
     - Log an MLFlow Run to record the work
     - Manage input parameters
     - Manage result data set
     - Manage data transfer between the caller and the unit of work (using Data Handlers)
    """
    mlflow_run = None
    result_data_handler = None
    handled_params = None
    log_result = None

    def __init__(self, run_id=None, result_data_handler=None,
                 handled_params=None, log_result=False,
                 experiment_name=None):

        if run_id is None:
            # No run ID given
            if len(ACTIVE_TASK_STACK) > 0:
                # Active Task Run Found, overriding TaskRun
                self_task = ACTIVE_TASK_STACK.pop()
                self.__dict__ = self_task.__dict__
                return None
            else:
                # Check environment variable
                active_run = mlflow.active_run()
                env_run_id = os.environ.get("MLFLOW_RUN_ID")
                if active_run is None and env_run_id is not None:
                    active_run = mlflow.get_run(env_run_id)
                if active_run:
                    # Run ID found in environment variable
                    self_task = TaskRun(run_id=active_run.info.run_id)
                    self.__dict__ = self_task.__dict__
                else:
                    # No run_id given and no MLFlow run found, creating new one
                    if experiment_name:
                        experiment = get_or_create_experiment(experiment_name)
                        experiment_id = experiment.experiment_id
                        mlflow_run = mlflow.start_run(nested=True, experiment_id=experiment_id)
                    else:
                        mlflow_run = mlflow.start_run(nested=True)
                    self.mlflow_run = mlflow_run
                    mlflow.end_run()
                    if result_data_handler is None:
                        # No result handler given, see if there is one logged
                        dh_path = "/".join([str(mlflow_run.info.run_id),
                                            "result"])
                        found_data_handler = data_handler_from_path(dh_path)
                        if found_data_handler is None:
                            # No result handler in log either, create a pickle handler
                            found_data_handler = data_handlers.PickleDataHandler(data_name="result")
                    self.result_data_handler = result_data_handler or found_data_handler
                    self.handled_params = handled_params
                    self.log_result = log_result
        else:
            # Run ID given, looking it up
            self.mlflow_run = mlflow.get_run(run_id)
            if result_data_handler is None:
                # No result handler given, see if there is one logged
                dh_path = "/".join([str(run_id),
                                    "result"])
                found_data_handler = data_handler_from_path(dh_path)
                if found_data_handler is None:
                    # No result handler in log either, create a pickle handler
                    found_data_handler = data_handlers.PickleDataHandler(data_name="result")
            self.result_data_handler = result_data_handler or found_data_handler
            self.handled_params = handled_params
            self.log_result = log_result

    def get_params(self):
        """
        Unpacks the task parameters stored in the `self.params` dictionary by resolving any `AbstractTask` instances.
        Iterates through each key-value pair in `self.params`, where keys represent parameter names and values represent parameter values. For any value that is an instance of `AbstractTask`, the function calls `get_result` on that task instance to retrieve its result. For values that are not `AbstractTask` instances, the function keeps the original value. The function returns a new dictionary where any `AbstractTask` instances are replaced by their results.

        Returns:
            dict: A new dictionary where any `AbstractTask` instance values in `self.params` are replaced by the results of calling `get_result` on them, while other key-value pairs remain as in `self.params`.
        """
        # Get param data handlers
        if self.handled_params is None:
            self.handled_params = {}

            log_params = self.mlflow_run.data.params

            # fetch param info from MLFlow
            for param_name, param_path in log_params.items():
                self.handled_params[param_name] = data_handler_from_path(param_path)

        # Convert data handlers to raw values
        param_values = {}
        for p, val in self.handled_params.items():
            param_values[p] = val.get()

        return param_values

    def set_result(self, result):
        """
        Sets and processes the result of the task.
        """

        self.result_data_handler.set(result)

    def get_result(self):
        """
        Retrieves the result of the task.
        """
        # Get result data handler
        if self.result_data_handler is None:
            # fetch result info from MLFlow
            dh_path = "/".join([self.mlflow_run.info.run_id,
                                "result"])
            self.result_data_handler = data_handler_from_path(dh_path)

        result = self.result_data_handler.get()
        return result


class AbstractTask(ABC):
    """
    Abstract class for Tasks: units of work in specific formats (scripts, endpoints, etc)
    and logging those units of work as Experiments in MLFlow

    Goals:
     - Abstract the physical format and execution methods of units of work
     - Parameterize the units of work for reuse
     - Log Experiments in MLFlow to document each unit of work
     - Expose a single Result data set from the unit of work
     - Manage dependencies and execution of the unit of work
     - Manage data transfer between the caller and the unit of work (using Data Handlers)

    Attributes:
       task_action   (any) Physical location of the unit of work (e.g. path to a script, uri for an endpoint)
       experiment    (mlflow Experiment) MLFlow Experiment object corresponding to this Task
       use_cache     (bool) Force caching of parameter inputs before running the Task
       autolog       (bool) Use MLFlow autologging
       result_data_handler (Data Handler) The data handler to be used for the result dataset
       default_data_handler (Data Handler) The data handler to be used for parameters
    """
    task_action = None
    experiment = None
    log_params = False
    log_result = False
    autolog = True
    result_data_handler_class = data_handlers.PickleDataHandler
    default_data_handler_class = data_handlers.PickleDataHandler

    def __init__(self, task_action=None,
                 experiment_name=None,
                 experiment_id=None,
                 log_params=False, log_result=False, autolog=True,
                 result_data_handler_class=None,
                 default_data_handler_class=None):
        """
        Initializes the task with an MLFlow "experiment"
        """
        self.ACTIVE_TASK_STACK = ACTIVE_TASK_STACK
        self.task_action = task_action
        self.log_params = log_params
        self.log_result = log_result
        self.autolog = autolog
        self.result_data_handler_class = result_data_handler_class or self.result_data_handler_class
        self.default_data_handler_class = default_data_handler_class or self.default_data_handler_class

        # Process config ##################
        #   Get/Create the experiment
        if experiment_id is not None:
            self.experiment = mlflow.get_experiment(experiment_id)
        elif experiment_name is not None:
            self.experiment = get_or_create_experiment(experiment_name)
        else:
            self.experiment = get_or_create_experiment(str(task_action))

    @abstractmethod
    def __exec_task__(self, task_action, params, run_id, task_run) -> str:
        """
        Run task format-specific execution to complete the task
        Task.set_result() must be called here or in the task process
        Returns task status code (e.g. "FINISHED")
        """
        pass

    #  Concrete method
    def run(self, **params):
        """
        Executes the task.
        """
        #   Start MLFlow Run
        mlflow_run = mlflow.start_run(experiment_id=self.experiment.experiment_id,
                                      nested=True)

        run_id = mlflow_run.info.run_id

        # Process params ##################
        handled_params = self.__handle_params__(params, run_id=run_id,
                                                log_full_values=self.log_params)

        # Set up result data handler
        task_result_handler = self.result_data_handler_class(data_name="result")
        task_result_handler.register(run_id=run_id)

        # Create the Task Run
        task_run = TaskRun(run_id, task_result_handler, handled_params)

        # Run the task ##################
        print(f"TASK: {self.experiment.name} RUNNING {self.experiment.experiment_id} / {run_id}")
        if self.autolog:
            mlflow.autolog()
        final_run_status = self.__exec_task__(self.task_action, handled_params,
                                              run_id, task_run)

        # Update status ##################
        mlflow.end_run(final_run_status)
        print(f"TASK: {self.experiment.name} {final_run_status} {self.experiment.experiment_id} / {run_id}")

        return task_run

    #  Concrete method
    def __handle_params__(self, params, run_id, log_full_values=False):
        """
        Logs parameters of the task.
        Returns string-only params, which are uris for non-string param values
        """
        logged_params = {}
        handled_params = {}

        # take non-string params, cache them and replace with uris
        for p, val in params.items():
            if isinstance(val, str):
                print(f"DEBUG param {p} is a string")
                # TODO This param is a string, it can be logged as-is?
                data_name = f"params/{p}"
                p_handler = self.default_data_handler_class(data_name=data_name)
                p_handler.register(run_id)
                p_handler.set(val)
                if log_full_values:
                    p_handler.log()
                logged_params[p] = f"{run_id}/{p_handler.data_name}"
                handled_params[p] = p_handler
            elif isinstance(val, AbstractDataHandler):
                print(f"DEBUG param {p} is a data handler")
                # This param is a Data Handler already
                p_handler = val
                if log_full_values:
                    p_handler.log()
                logged_params[p] = f"{p_handler.mlflow_run.info.run_id}/{p_handler.data_name}"
                handled_params[p] = p_handler
            elif isinstance(val, TaskRun):
                print(f"DEBUG param {p} is a task")
                # This param is a Task, we need to get its result handler
                p_handler = val.result_data_handler
                if log_full_values:
                    p_handler.log()
                logged_params[p] = f"{p_handler.mlflow_run.info.run_id}/{p_handler.data_name}"
                handled_params[p] = p_handler
            else:
                print(f"DEBUG param {p} is something else")
                data_name = f"params/{p}"
                p_handler = self.default_data_handler_class(data_name=data_name)
                p_handler.register(run_id)
                p_handler.set(val)
                if log_full_values:
                    p_handler.log()
                logged_params[p] = f"{run_id}/{p_handler.data_name}"
                handled_params[p] = p_handler

        mlflow.log_params(logged_params)

        return handled_params