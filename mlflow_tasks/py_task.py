from mlflow_tasks.abstract_task import AbstractTask


class PyTask(AbstractTask):

    def __exec_task__(self, task_action, params, run_id, task_run):
        # Run the script with access to the local variable
        with open(task_action) as script_file:
            script_code = script_file.read()
            self.ACTIVE_TASK_STACK.append(task_run)
            exec(script_code, {}, {"ACTIVE_TASK_STACK": self.ACTIVE_TASK_STACK})

        # End the run
        return "FINISHED"