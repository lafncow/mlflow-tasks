import mlflow
import os
import subprocess
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import nbformat
from nbconvert import HTMLExporter
import papermill
import shutil
from typing import Callable
from mlflow.entities import RunStatus
from . import data_handlers

active_data_handlers = {
    "default": data_handlers.Py_Obj_Handler,
    "py_obj": data_handlers.Py_Obj_Handler
}

# TODO make this adjustable, maybe with env variable?
cache_dir = os.path.join(os.path.abspath(''), "mlflow_tasks_cache")

def data_handler_from_path(full_path):
    import yaml
    mlflow_client = MlflowClient()
    metadata_uri = full_path + "_meta.yaml"
    run_id = metadata_uri.split("/")[1]
    experiment_id = metadata_uri.split("/")[0]
    metadata_path = "/".join(metadata_uri.split("/")[2:])
    # Download metadata from log
    try:
        local_metadata_uri = mlflow_client.download_artifacts(run_id, metadata_path)
    except:
        raise Exception(f"No metadata found at {metadata_path}. Could not get data handler for {full_path}.")
    # Read the metadata
    with open(local_metadata_uri, 'r') as metadata_file:
        metadata = yaml.safe_load(metadata_file)
    # Find the right data handler
    data_handler_name = metadata['data_handler']
    if not data_handler_name in data_handlers.__dict__:
        raise Exception(f"Data handler {data_handler_name} not found.")
        return None
    # Create data handler
    data_handler = data_handlers.__dict__[data_handler_name](**metadata['handler_args'])
    # Load the data
    data_handler.load(metadata['full_path'])
    
    return data_handler

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

def register_data_handler(handler_name, handler):
    active_data_handlers[handler_name] = handler

class Task:
    def __init__(self, action=None, run_id=None, experiment_id=None, experiment_name=None, write_log=False, write_local_cache=False, write_global_cache=False, autolog=True, log_nb_html=True, data_handler=None, **params):
        self.result = None
        self.run = None
        self.write_log = write_log
        self.write_local_cache = write_local_cache
        self.write_global_cache = write_global_cache
        self.log_nb_html = log_nb_html
        self.params = params
        
        if "FLOW_LOG_RESULT" in os.environ:
            if (os.environ["FLOW_LOG_RESULT"] == "True") or (os.environ["FLOW_LOG_RESULT"] == "TRUE"):
                self.write_log = True

        if "FLOW_LOG_NB_HTML" in os.environ:
            if (os.environ["FLOW_LOG_NB_HTML"] == "True") or (os.environ["FLOW_LOG_NB_HTML"] == "TRUE"):
                self.log_nb_html = True

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
        
        if data_handler is None:
            self.data_handler = active_data_handlers['default'](self.experiment_id, self.run_id, "result")
        else:
            self.data_handler = data_handler
        
        self.print_status()
        
        if autolog:
            mlflow.autolog()
            self.autolog = True

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
        clean_params = self.__log_params__(cache_local=True)
        
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
        
    def __exec_nb__(self, nb_path):
        # TODO add nb_path to run information
        # Log params
        clean_params = self.__log_params__(cache_local=True)
        
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
        self.params = {"model_uri": model_uri, "model_input": model_input}
        clean_params = self.__log_params__()

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
                    p_handler.cache_local()
                if cache_global:
                    p_handler.cache_global()
                if write_log:
                    p_handler.log()
                params_as_strs[p] = p_handler.full_path
                
            else:
                sub_path = "/".join(["params", p])
                p_handler = active_data_handlers['default']()
                p_handler.set(val, self.experiment_id, self.run_id, sub_path)
                
                if cache_local:
                    p_handler.cache_local()
                if cache_global:
                    p_handler.cache_global()
                if write_log:
                    p_handler.log()
                params_as_strs[p] = p_handler.full_path

        mlflow.log_params(params_as_strs)

        return params_as_strs

    def get_run(self):
        self.run = mlflow.get_run(self.run.info.run_id)
        return self.run
    
    def set_result(self, result):
        
        self.result = result
        self.data_handler.set(result, self.experiment_id, self.run_id, "result")
            
        if self.write_local_cache:
            self.data_handler.cache_local()
        if self.write_global_cache:
            self.data_handler.cache_global()
        if self.write_log:
            self.data_handler.log()
        
        return self.data_handler
    
    def get_result(self):
        if self.data_handler is None:
            self.data_handler = data_handler_from_path("/".join([self.experiment_id, self.run_id, "result"]))
        result = self.data_handler.get()
        if result is None:
            self.data_handler = data_handler_from_path("/".join([self.experiment_id, self.run_id, "result"]))
            result = self.data_handler.get()
        return result
    
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
        if not self.params is None:
            # Unpack Task params
            unpacked_params = {}
            for p, val in self.params.items():
                if isinstance(val, Task):
                    unpacked_params[p] = val.get_result()
                else:
                    unpacked_params[p] = val
            return unpacked_params

        self.get_run()
        params = self.run.data.params or {}

        # Add any given params
        if not default_params is None:
            default_params.update(params)
            params = default_params

        for p_name, p_val in params.items():
            # Load param data handlers
            param_handler = data_handler_from_path(p_val)
            if not param_handler is None:

                # Update with rehydrated values
                p_val = param_handler.get()
                params[p_name] = p_val
        
        self.params = params

        return self.params

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
