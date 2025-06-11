import os
import tempfile
from mlflow_tasks.abstract_task import AbstractTask
import mlflow
from nbconvert import HTMLExporter
import papermill


class NotebookTask(AbstractTask):

    def __exec_task__(self, task_action, params, run_id, task_run):
        # Set MLFlow environment variables
        env_old = os.environ.copy()
        os.environ["MLFLOW_TRACKING_URI"] = mlflow.tracking.get_tracking_uri()
        os.environ["MLFLOW_EXPERIMENT_NAME"] = self.experiment.name
        os.environ["MLFLOW_RUN_ID"] = str(run_id)

        # Cache params
        for p_name, p_handler in params.items():
            p_handler.cache()

        # Make sure result caches
        task_run.result_data_handler.use_log = True
        task_run.result_data_handler.__log_metadata__()

        # Save the processed notebook to a temp location
        nb_name = os.path.splitext(os.path.split(task_action)[1])[0]
        nb_result_name = nb_name+"_result.ipynb"
        temp_dir = tempfile.TemporaryDirectory()
        nb_result_path = os.path.join(temp_dir.name, nb_result_name)

        # Parameter paths passed as parameters only for documentation in the notebook
        # Complex data structures can't be passed through papermill
        # Instead, the real parameters are passed through caching and the TaskRun
        param_paths = {key: val.data_name for key, val in params.items()}

        # Run the notebook
        papermill.execute_notebook(
           task_action,
           nb_result_path,
           parameters=param_paths
        )

        # Export the notebook to HTML and return the path
        html_path = nb_result_path.split(".")[0] + ".html"
        # Create HTML exporter
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        (html_text, resources) = html_exporter.from_filename(nb_result_path)
        html_file = open(html_path, 'wb')
        html_file.write(html_text.encode("utf-8"))
        html_file.close()

        # Log the processed notebook as HTML
        mlflow.log_artifact(html_path)

        # Reset the MLFlow environment variables
        os.environ = env_old

        # End the run
        return "FINISHED"