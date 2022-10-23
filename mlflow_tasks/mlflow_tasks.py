import mlflow
import os
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
from nbconvert import HTMLExporter
import papermill
from typing import Callable
from mlflow.entities import RunStatus
from . import data_handlers
from .data_handlers.utility import cache_dir, data_handler_from_path

default_data_handler = data_handlers.Py_Obj_Handler

special_task_params = ['write_log', 'write_local_cache', 'write_global_cache', 'autolog']

active_task_stack = []

def get_or_create_experiment(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment {experiment_name} does not exist on MLFlow yet, creating it...")
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment(experiment_id)
    
    return experiment

def active_task():
    active_run = mlflow.active_run()
    if active_run:
        return Task(run_id=active_run.info.run_id)
    else:
        return None

def get_task(run_id):
    return Task(run_id=run_id)

def start_task(**args):
    return Task(**args)

class Task:
    def __init__(self, action=None, run_id=None, experiment_id=None, experiment_name=None, write_log=False, write_local_cache=False, write_global_cache=False, autolog=True, data_handler=None, **params):
        # If there is an active Task, then this is being run in a script Task
        if len(active_task_stack) > 0:
            # Pop the active Task off of the stack and assume it
            # Note arguments to Task() inside the script are being ignored
            self_task = active_task_stack.pop()
            self.__dict__ = self_task.__dict__
            return None
            
        self.result = None
        self.run = None
        self.write_log = write_log
        self.write_local_cache = write_local_cache
        self.write_global_cache = write_global_cache
        self.params = params
        self.autolog = autolog

        action_name = None

        if not action is None:
            # It is a function
            if isinstance(action, Callable):
                action_name = action.__name__
            # It is a string
            elif isinstance(action, str):
                action_name = os.path.basename(action)
                if action[-6:] == ".ipynb":
                    self.write_local_cache = True
                if (action_name is None) or (action_name == ""):
                    # May be needed for running model uris
                    print(f"DEBUG: needed to set action_name differently for model URI {action} -> {os.path.split(action)[-1]}")
                    action_name = os.path.split(action)[-1]

        if experiment_name is None:
            experiment_name = action_name

        ## Get/Create the experiment
        if not experiment_id is None:
            self.experiment_id = experiment_id
        elif not experiment_name is None:
            task_experiment = get_or_create_experiment(experiment_name)
            self.experiment_id = task_experiment.experiment_id
        else:
            self.experiment_id = None
        
        ## Start MLFlow Run
        self.run = mlflow.start_run(run_id=run_id, experiment_id=self.experiment_id, nested=True)
        
        self.run_id = self.run.info.run_id
        self.experiment_id = self.run.info.experiment_id
        self.experiment_name = mlflow.get_experiment(self.run.info.experiment_id).name

        ## Load task params
        for p in special_task_params:
            if p in self.run.data.tags:
                param_val = self.run.data.tags[p]
                if param_val == "True":
                    param_val = True
                elif param_val == "False":
                    param_val = False
                self.__setattr__(p, param_val)

        ## Create the data handler
        if data_handler is None:
            self.data_handler = default_data_handler()
        else:
            self.data_handler = data_handler
        self.data_handler.register(self.experiment_id, self.run_id, "result")

        ## Save task params
        for p in special_task_params:
            param_val = self.__getattribute__(p)
            if not param_val is None:
                mlflow.set_tag(p, param_val)
        
        self.print_status()
        
        if self.autolog:
            mlflow.autolog()

        ## Execute the action for the task
        if not action is None:
            end_status = "FAILED"
            # It is a function
            if isinstance(action, Callable):
                end_status = self.__exec_func__(action)
            # It is a string
            elif isinstance(action, str):
                # ends in .py
                if action[-3:] == ".py":
                    end_status = self.__exec_script__(action)
                # ends in .ipynb
                elif action[-6:] == ".ipynb":
                    end_status = self.__exec_nb__(action)
                elif "model_input" in params:
                # is model uri
                # TODO validate model uri
                    end_status = self.__exec_model__(action, params["model_input"])
            else:
                #Fail
                raise Exception("Invalid task action type (not a function or string).")
                
            # End the task
            self.end_run(end_status)

    def __enter__(self):
        # Allow this to be a context manager
        return self

    def __exit__(self, type, value, traceback):
        self.end_run()

    def __exec_func__(self, func):
        # TODO add func to run information
        # Log params
        self.__log_params__()
        
        # Unpack Task params
        unpacked_params = {}
        for p, val in self.params.items():
            if isinstance(val, Task):
                unpacked_params[p] = val.get_result()
            else:
                unpacked_params[p] = val
        
        # Run the task
        result_data = func(**unpacked_params)
        
        # Save result
        self.set_result(result_data)

        # End the run
        return "FINISHED"

    def __exec_script__(self, script_path):
        # Log params
        # TODO add script_path to run information
        clean_params = self.__log_params__()
        
        # Save self task to a local variable
        active_task_stack.append(self)
        loc_vars = {"active_task_stack": active_task_stack}
        # Run the script with access to the local variable
        with open(script_path) as script_file:
            script_code = script_file.read()
            exec(script_code, {}, loc_vars)
        
        # End the run
        return "FINISHED"
        
    def __exec_nb__(self, nb_path):
        # TODO add nb_path to run information
        # Log params
        clean_params = self.__log_params__(cache_local=True)
        
        # Set MLFlow environment variables
        env_old = os.environ.copy()
        os.environ["MLFLOW_TRACKING_URI"] = mlflow.tracking.get_tracking_uri()
        os.environ["MLFLOW_EXPERIMENT_NAME"] = self.experiment_name
        os.environ["MLFLOW_RUN_ID"] = str(self.run_id)

        nb_name = os.path.splitext(os.path.split(nb_path)[1])[0]
        nb_result_name = nb_name+"_result.ipynb"
        nb_result_path = os.path.join(cache_dir, self.experiment_id, self.run_id, "artifacts", nb_result_name)
        os.makedirs(os.path.join(cache_dir, self.experiment_id, self.run_id, "artifacts"), exist_ok=True)
        
        # Run the notebook
        papermill.execute_notebook(
           nb_path,
           nb_result_path,
           parameters = clean_params
        )
        
        # Reset the MLFlow environment variables
        os.environ = env_old
        
        """Export the notebook to HTML and return the path"""
        html_path = nb_result_path.split(".")[0] + ".html"
        # Create HTML exporter
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        (html_text, resources) = html_exporter.from_filename(nb_result_path)
        html_file = open(html_path,'wb')
        html_file.write(html_text.encode("utf-8"))
        html_file.close()
        mlflow.log_artifact(html_path)
        #mlflow.log_text(html_text.encode("utf-8"), html_path)
        print(f"HTML Report was generated: {html_path}")
        
        # End the run
        return "FINISHED"

    def __exec_model__(self, model_uri, model_input):
        # TODO change they way model_uri is recorded on the run?
        # Log params      
        clean_params = self.__log_params__()
        
        # Unpack Task params
        if isinstance(model_input, Task):
            model_input = model_input.get_result()

        # Run the task
        model = mlflow.pyfunc.load_model(model_uri)
        result_data = model.predict(model_input)
        
        # Save result
        self.set_result(result_data)

        # End the run
        return "FINISHED"

    def __log_params__(self, cache_local=False, cache_global=False, write_log=False):

        params_as_strs = {}

        # take non-string params, cache them and replace with uris
        for p, val in self.params.items():
            if isinstance(val, str):
                params_as_strs[p] = val

            elif isinstance(val, Task):
                p_handler = val.data_handler
                if cache_local:
                    p_handler.get()
                    p_handler.cache_local()
                if cache_global:
                    p_handler.get()
                    p_handler.cache_global()
                if write_log:
                    p_handler.get()
                    p_handler.log()
                params_as_strs[p] = p_handler.full_path
                
            else:
                sub_path = "/".join(["params", p])
                p_handler = default_data_handler()
                p_handler.register(self.experiment_id, self.run_id, sub_path)
                p_handler.set(val)
                
                if cache_local:
                    p_handler.cache_local()
                if cache_global:
                    p_handler.cache_global()
                if write_log:
                    p_handler.log()
                params_as_strs[p] = p_handler.full_path

        mlflow.log_params(params_as_strs)
        self.get_run()

        return params_as_strs

    def get_run(self):
        self.run = mlflow.get_run(self.run.info.run_id)
        return self.run
    
    def set_result(self, result):
        
        self.result = result
        self.data_handler.set(result)
            
        if self.write_local_cache:
            self.data_handler.cache_local()
        if self.write_global_cache:
            self.data_handler.cache_global()
        if self.write_log:
            self.data_handler.log()
        
        return self.data_handler
    
    def get_result(self):
        result = self.data_handler.get()
        return result
    
    def get_params(self):
        # Collect all params from log and combine with params passed to Task()
        logged_param_strings = self.run.data.params
        for key, val in logged_param_strings.items():
            # Check if we have it
            if not key in self.params:
                # Load param data handlers
                param_handler = data_handler_from_path(val)
                if not param_handler is None:
                    # Unpack value
                    self.params[key] = param_handler.get()
                else:
                    # No data handler needed
                    self.params[key] = val

        # Upack any passed params that are tasks
        for key, val in self.params.items():
            if isinstance(val, Task):
                    self.params[key] = val.get_result() 

        return self.params

    def print_status(self):
        status = self.get_run().info.status
        print(f"TASK: {self.experiment_name} {status} {self.experiment_id} / {self.run_id}")
        
    def end_run(self, status="FINISHED"):
        #End the run
        mlflow.end_run(status)
        self.print_status()

class Flow(Task):
    
    def end_flow(self):
        #End the run
        self.end_run()
    
    def start_task(self, *args, **kwargs):
        return Task(*args, **kwargs)
