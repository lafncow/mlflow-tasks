from mlflow_tasks.abstract_task import AbstractTask


class FuncTask(AbstractTask):

    def __exec_task__(self, task_action, params, run_id, task_run):
        # Unpack Task params
        unpacked_params = {}
        for p, val in params.items():
            unpacked_params[p] = val.get()

        # Run the task
        result_data = task_action(**unpacked_params)

        # Save result
        task_run.set_result(result_data)

        # End the run
        return "FINISHED"