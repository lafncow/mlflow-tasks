# +
import os
import sys

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

test_tracking_folder = os.path.join(project_folder, "test_mlruns")

# Set the mlflow tracking folder
mlflow.set_tracking_uri(f"file:/{test_tracking_folder}")


# +
# Test Utility Functions
def test_create_flow():
    flow = mlflow_tasks.Flow(experiment_name="Flow Test Experiment")
    flow.end_flow()
    assert isinstance(flow, mlflow_tasks.Flow)

def test_create_flow_named():
    flow = mlflow_tasks.Flow(experiment_name="Flow Test Experiment")
    flow.end_flow()
    assert flow.experiment_name == "Flow Test Experiment"

def test_create_end_flow():
    
    flow = mlflow_tasks.Flow(experiment_name="Flow Test Experiment")
    flow.end_flow()
    assert flow.get_run().info.status == "FINISHED"
