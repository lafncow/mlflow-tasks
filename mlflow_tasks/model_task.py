from mlflow_tasks.abstract_task import AbstractTask
import mlflow


class ModelTask(AbstractTask):

    def __exec_task__(self, task_action, params, run_id, task_run):

        # Models can have only one input
        # Grabbing the first input param
        # Additional input params will be ignored
        model_input = list(params.items())[0][1].get()

        # Run the task
        model = mlflow.pyfunc.load_model(task_action)
        result_data = model.predict(model_input)
        print(result_data)

        # Save result
        task_run.set_result(result_data)

        # End the run
        return "FINISHED"