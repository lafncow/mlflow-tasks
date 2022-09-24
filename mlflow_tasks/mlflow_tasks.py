import mlflow
import os
import subprocess
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import pickle
import nbformat
from nbconvert import HTMLExporter
import papermill
import shutil
from typing import Callable
from mlflow.entities import RunStatus
from . import data_handlers

default_handler = data_handlers.Py_Obj_Handler()
pandas_df_handler = data_handlers.Pandas_Df_Handler()

active_data_handlers = {
    "default": default_handler,
    "py_obj": default_handler,
    "pandas": pandas_df_handler
}

mlflow_client = MlflowClient()
# TODO make this adjustable, maybe with env variable?
cache_dir = os.path.join(os.path.abspath(''), "mlflow_tasks_cache")

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

def set_data_handler(handler_name, handler):
    active_data_handlers[handler_name] = handler

class Task:
    def __init__(self, action=None, run_id=None, experiment_id=None, experiment_name=None, write_log=True, autolog=True, log_nb_html=True, data_handler="default", **params):
        self.result = None
        self.run = None
        self.log_uri = None
        self.cache_uri = None
        self.write_log = write_log
        self.log_nb_html = log_nb_html
        
        if "FLOW_LOG_RESULT" in os.environ:
            if (os.environ["FLOW_LOG_RESULT"] == "True") or (os.environ["FLOW_LOG_RESULT"] == "TRUE"):
                self.write_log = True

        if "FLOW_LOG_NB_HTML" in os.environ:
            if (os.environ["FLOW_LOG_NB_HTML"] == "True") or (os.environ["FLOW_LOG_NB_HTML"] == "TRUE"):
                self.log_nb_html = True

        if data_handler in active_data_handlers:
            self.data_handler = active_data_handlers[data_handler]
        else:
            self.data_handler = active_data_handlers['default']

        action_name = None

        if not action is None:
            # It is a function
            if isinstance(action, Callable):
                action_name = action.__name__
            # It is a string
            elif isinstance(action, str):
                action_name = os.path.basename(action)
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
        
        self.print_status()
        
        if autolog:
            mlflow.autolog()
            self.autolog = True

        if not action is None:
            end_status = "FAILED"
            # It is a function
            if isinstance(action, Callable):
                end_status = self.__exec_func__(action, **params)
            # It is a string
            elif isinstance(action, str):
                # ends in .py
                if action[-3:] == ".py":
                    end_status = self.__exec_script__(action, **params)
                # ends in .ipynb
                elif action[-6:] == ".ipynb":
                    end_status = self.__exec_nb__(action, **params)
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

    def __exec_func__(self, func, **params):
        # TODO add func to run information
        # Log params
        self.__log_params__(params)
        
        # Run the task
        result_data = func(**params)
        
        # Save result
        self.set_result(result_data)

        # End the run
        return "FINISHED"

    def __exec_script__(self, script_path, **params):
        # Log params
        # TODO add script_path to run information
        clean_params = self.__log_params__(params)
        
        # Set MLFlow environment variables
        new_env = os.environ.copy()
        new_env.update({
            "MLFLOW_TRACKING_URI": mlflow.tracking.get_tracking_uri(),
            "MLFLOW_EXPERIMENT_NAME": self.experiment_name,
            "MLFLOW_RUN_ID": str(self.run_id),
            "FLOW_LOG_RESULT": str(self.write_log)
        })
        
        # Convert parameters into commandline arguments
        args = ["python", script_path]
        for key, val in clean_params.items():
            args.append(f"-{key}")
            args.append(str(val))
        
        # Run the task as a subprocess
        process_res = subprocess.run(args, env=new_env, capture_output=True)
        
        print(process_res.stdout)
        
        # Check if task was successful
        if process_res.returncode != 0:
            # End the run
            print(process_res.stderr)
            return "FAILED"
        else:
            # End the run
            return "FINISHED"
        
    def __exec_nb__(self, nb_path, **params):
        # TODO add nb_path to run information
        # Log params
        clean_params = self.__log_params__(params)
        
        # Set MLFlow environment variables
        env_old = os.environ.copy()
        os.environ["MLFLOW_TRACKING_URI"] = mlflow.tracking.get_tracking_uri()
        os.environ["MLFLOW_EXPERIMENT_NAME"] = self.experiment_name
        os.environ["MLFLOW_RUN_ID"] = str(self.run_id)
        os.environ["FLOW_LOG_RESULT"] = str(self.write_log)

        nb_name = os.path.splitext(os.path.split(nb_path)[1])[0]
        nb_result_name = nb_name+"_result.ipynb"
        nb_result_path = os.path.join(cache_dir, self.experiment_id, self.run_id, "artifacts", nb_result_name)
        
        # Run the notebook
        papermill.execute_notebook(
           nb_path,
           nb_result_path,
           parameters = clean_params
        )
        
        # Reset the MLFlow environment variables
        os.environ = env_old
        
        if self.log_nb_html:
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
        params = {"model_uri": model_uri, "model_input": model_input}
        clean_params = self.__log_params__(params)

        # Run the task
        model = mlflow.pyfunc.load_model(model_uri)
        result_data = model.predict(model_input)
        
        # Save result
        self.set_result(result_data)

        # End the run
        return "FINISHED"

    def __log_params__(self, params):

        clean_params = {}

        # take non-string params, cache them and replace with uris
        for p, val in params.items():
            if isinstance(val, str):
                clean_params[p] = val
            elif isinstance(val, Task):
                val.get_cache_uri():
                # Create copy from cache
                param_link_uri = os.path.join(cache_dir, self.experiment_id, self.run_id, "artifacts", "params", p)
                os.makedirs(param_link_uri)
                #shutil.copyfile(val.cache_uri, param_link_uri)
                from distutils.dir_util import copy_tree
                copy_tree(val.cache_uri, param_link_uri)
                clean_params[p] = param_link_uri
            else:
                p_path = os.path.join("params", p)
                uri = self.cache_artifact(val, p_path)
                if self.write_log:
                    uri = mlflow.log_artifact(uri, p_path)
                clean_params[p] = uri

        mlflow.log_params(clean_params)

        return clean_params

    def get_run(self):
        self.run = mlflow.get_run(self.run.info.run_id)
        return self.run
    
    def set_result(self, result, write_log=False):
        
        self.result = result
            
        if write_log or self.write_log:
            self.log_result(result)
        else:
            self.cache_result(result)
        
        return True
    
    def get_result(self):
        # Check run status and warn/fail if not complete
        if self.get_run().info.status != "FINISHED":
            print(f"Can't get results, the run is in status {self.run.info.status}")
            return None
        
        # Return it if we already have it
        if not self.result is None:
            return self.result
        
        # Return it if it is cached
        if self.cache_uri is None:
            self.cache_uri = self.get_cache_uri()
        
        if not self.cache_uri is None:
            # Unpack the data into a variable
            self.result = self.data_handler.load(self.cache_uri)
            
            return self.result
        
        # Return it if it is logged
        if self.log_uri is None:
            self.log_uri = self.get_log_uri()
        
        if not self.log_uri is None:
            download_dir = os.path.join(cache_dir, task_experiment.experiment_id, task_run.info.run_id, "result")
            # Download data from uri
            self.cache_uri = mlflow_client.download_artifacts(self.log_uri, "result", download_dir)
        
            # Unpack the data into a variable
            self.result = self.data_handler.load(cache_uri)
            
            return self.result
        
        # Data not found!
        return None
    
    def log_result(self, result):
        self.result = result
        
        # Save result to a file
        cache_uri = self.cache_result(result)

        # Log result to MLFlow
        mlflow.log_artifact(cache_uri)
        
        log_uri = self.get_log_uri()
        self.log_uri = log_uri
        
        return log_uri
    
    def cache_result(self, result):
        self.result = result
        
        result_dir = os.path.join(cache_dir, self.run.info.experiment_id, self.run.info.run_id, "result")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        cache_uri = self.data_handler.cache(result, result_dir)

        self.cache_uri = cache_uri
        
        return cache_uri
    
    def cache_artifact(self, val, artifact_path, data_handler="default"):
        
        # TODO make internal method
        
        artifact_cache_dir = os.path.join(cache_dir, self.run.info.experiment_id, self.run.info.run_id, "artifacts")
        artifact_cache_dir = os.path.join(artifact_cache_dir, artifact_path)
        if not os.path.exists(artifact_cache_dir):
            os.makedirs(artifact_cache_dir)
        cache_uri = active_data_handlers[data_handler].cache(val, artifact_cache_dir)
        
        return cache_uri
    
    def get_params(self, default_params=None):
        self.get_run()
        params = self.run.data.params or {}
        
        # Add any given params
        if not default_params is None:
            default_params.update(params)
            params = default_params
        
        # List folders in the params cache directory
        params_cache_dir = os.path.join(cache_dir, self.experiment_id, self.run_id, "artifacts", "params")
        cache_dir_list = []
        if os.path.exists(params_cache_dir):
            cache_dir_list = os.listdir(params_cache_dir)
            # Filer to folders
            cache_dir_list = [os.path.join(params_cache_dir,f) for f in cache_dir_list if os.path.isdir(os.path.join(params_cache_dir,f))]

        # rehydrate params
        for cache_dir in cache_dir_list:
            param_name = os.path.split(cache_dir)[1]
            # Read the metadata
            import yaml
            metadata_file_name = os.path.join(cache_dir, "meta.yaml")
            with open(metadata_file_name, 'r') as metadata_file:
                metadata = yaml.safe_load(metadata_file)
            # Get the right data handler
            data_handler_args = None
            if "args" in metadata:
                data_handler_args = metadata["args"]
            data_handler_class = data_handlers[metadata["data_handler"]]
            data_handler = data_handler_class(**data_handler_args)
            # Unpack the data into a variable
            params[param_name] = data_handler.load(cache_dir)
        
        return params

    def get_cache_uri(self, artifact_path="result"):
        if self.cache_uri:
            return self.cache_uri
        
        cache_uri = None
        # List files in the result cache directory
        cache_uri = os.path.join(cache_dir, self.experiment_id, self.run_id, artifact_path)
            
        self.cache_uri = cache_uri

        return cache_uri

    def get_log_uri(self):
        if self.log_uri:
            return self.log_uri
        
        log_uri = None
        # List files in the log result directory
        log_uri = mlflow.get_artifact_uri("result")
            
        self.log_uri = log_uri

        return log_uri
    
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
