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

mlflow_client = MlflowClient()
# TODO make this adjustable, maybe with env variable?
cache_dir = os.path.join(os.path.split(os.path.abspath(''))[0], "tmp", "results_cache")

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
    def __init__(self, action=None, run_id=None, run_name=None, experiment_id=None, experiment_name=None, tags=None, write_log=True, autolog=True, log_nb_html=True, **params):
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
        self.run = mlflow.start_run(run_id=run_id, experiment_id=self.experiment_id, run_name=run_name, nested=True, tags=tags)
        
        self.run_id = self.run.info.run_id
        self.experiment_id = self.run.info.experiment_id
        if "mlflow.runName" in self.run.data.tags:
            self.run_name = self.run.data.tags["mlflow.runName"]
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
        
        # Run the notebook
        papermill.execute_notebook(
           nb_path,
           nb_path,
           parameters = clean_params
        )
        
        # Reset the MLFlow environment variables
        os.environ = env_old
        
        if self.log_nb_html:
            """Export the notebook to HTML and return the path"""
            html_path = nb_path.split(".")[0] + ".html"
            # Create HTML exporter
            html_exporter = HTMLExporter()
            html_exporter.template_name = 'classic'
            (html_text, resources) = html_exporter.from_filename(nb_path)
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
                if not val.result is None:
                    # Cache it
                    p_path = os.path.join("params", p)
                    uri = self.cache_artifact(val.result, p_path)
                    if self.write_log:
                        uri = mlflow.log_artifact(uri, p_path)
                    clean_params[p] = uri
                elif val.get_cache_uri():
                    # Create copy from cache
                    param_link_uri = os.path.join(cache_dir, self.experiment_id, self.run_id, "artifacts", "params", p)
                    if not os.path.exists(os.path.split(param_link_uri)[0]):
                        os.makedirs(os.path.split(param_link_uri)[0])
                    shutil.copyfile(val.cache_uri, param_link_uri)
                    clean_params[p] = param_link_uri
                elif not val.get_result() is None:
                    # Pull from log
                    p_path = os.path.join("params", p)
                    uri = self.cache_artifact(val.result, p_path)
                    if self.write_log:
                        uri = mlflow.log_artifact(uri, p_path)
                    clean_params[p] = uri
                else:
                    # Give up
                    print(f"No result value found on task passed as parameter! (Task {val.experiment_id} / {val.run_id})")
                    clean_params[p] = None
                    
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
            cache_file = open(self.cache_uri, 'rb')
            self.result = pickle.load(cache_file)
            cache_file.close()
            
            return self.result
        
        # Return it if it is logged
        if self.log_uri is None:
            self.log_uri = self.get_log_uri()
        
        if not self.log_uri is None:
            download_dir = os.path.join(cache_dir, task_experiment.experiment_id, task_run.info.run_id, "result")
            # Download data from uri
            self.cache_uri = mlflow_client.download_artifacts(self.log_uri, "result", download_dir)
        
            # Unpack the data into a variable
            cache_file = open(self.cache_uri, 'rb')
            self.result = pickle.load(cache_file)
            cache_file.close()
            
            return self.result
        
        # Data not found!
        return None
    
    def log_result(self, result):
        self.result = result
        
        # Save result to a file
        cache_uri = self.cache_result(result)

        # Log result to MLFlow
        mlflow.log_artifact(cache_uri, "result")
        
        log_uri = self.get_log_uri()
        self.log_uri = log_uri
        
        return log_uri
    
    def cache_result(self, result):
        self.result = result
        
        # TODO detect result type
        # TODO change result logging method based on type
        
        result_dir = os.path.join(cache_dir, self.run.info.experiment_id, self.run.info.run_id, "result")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        cache_uri = os.path.join(result_dir, "result.pkl")
        cache_file = open(cache_uri,'wb')
        pickle.dump(self.result, cache_file)
        cache_file.close()
        
        self.cache_uri = cache_uri
        
        return cache_uri
    
    def cache_artifact(self, val, artifact_path):
        
        # TODO make internal method
        # TODO detect result type
        # TODO change result logging method based on type
        
        artifact_cache_dir = os.path.join(cache_dir, self.run.info.experiment_id, self.run.info.run_id, "artifacts")
        #os.makedirs(artifact_cache_dir)
        cache_uri = os.path.join(artifact_cache_dir, artifact_path)
        if not os.path.exists(os.path.split(cache_uri)[0]):
            os.makedirs(os.path.split(cache_uri)[0])
        cache_file = open(cache_uri,'wb')
        pickle.dump(val, cache_file)
        cache_file.close()
        
        return cache_uri
    
    def get_params(self, default_params=None):
        self.get_run()
        params = self.run.data.params or {}
        
        # Add any given params
        if not default_params is None:
            default_params.update(params)
            params = default_params
        
        # List files in the params cache directory
        params_cache_dir = os.path.join(cache_dir, self.experiment_id, self.run_id, "artifacts", "params")
        cache_file_list = []
        if os.path.exists(params_cache_dir):
            cache_file_list = os.listdir(params_cache_dir)
            # Filer out folders
            cache_file_list = [os.path.join(params_cache_dir,f) for f in cache_file_list if os.path.isfile(os.path.join(params_cache_dir,f))]
        
        if (len(cache_file_list) == 0):
            if not os.path.exists(params_cache_dir):
                os.makedirs(params_cache_dir)
            # no cache files found, download any from the log
            try:
                params_cache_dir = mlflow_client.download_artifacts(self.run.info.run_id, "params", params_cache_dir)
                cache_file_list = os.listdir(params_cache_dir)
                # Filter out folders
                cache_file_list = [os.path.join(params_cache_dir,f) for f in cache_file_list if os.path.isfile(os.path.join(params_cache_dir,f))]
            except Exception as e:
                pass

        # rehydrate params
        for cache_uri in cache_file_list:
            param_name = os.path.split(cache_uri)[1]
            # Unpack the data into a variable
            cache_file = open(cache_uri, 'rb')
            params[param_name] = pickle.load(cache_file)
            cache_file.close()
        
        return params

    def get_cache_uri(self, artifact_path=None):
        if self.cache_uri:
            return self.cache_uri
        
        cache_uri = None
        # List files in the result cache directory
        task_cache_dir = os.path.join(cache_dir, self.experiment_id, self.run_id, "result")
        try:
            file_list = os.listdir(task_cache_dir)
        except:
            return None
        file_list = [os.path.join(task_cache_dir,f) for f in file_list if os.path.isfile(os.path.join(task_cache_dir,f))]

        if len(file_list) == 1:
            cache_uri = file_list[0]
        elif len(file_list) > 1:
            cache_uri = file_list[0]
            print(f"Found multiple result files found in cache! Taking the first one: {cache_uri}")
        else:
            print("No result files found in cache")
            
        self.cache_uri = cache_uri

        return cache_uri

    def get_log_uri(self):
        if self.log_uri:
            return self.log_uri
        
        log_uri = None
        # List files in the log result directory
        file_list = mlflow_client.list_artifacts(self.run_id, "result")

        if len(file_list) == 1:
            log_uri = file_list[0]
        elif len(file_list) > 1:
            log_uri = file_list[0]
            print(f"Found multiple result files in log! Taking the first one: {log_uri}")
        else:
            print("No result files found in log")
            
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
