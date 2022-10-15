from mlflow_tasks.data_handlers import path_to_dir_uri
from mlflow_tasks.data_handlers import path_to_metadata_dir_uri
from mlflow_tasks.data_handlers import path_to_exp_run_path


def test_path_to_dir_uri():
    local_dir, local_uri = path_to_dir_uri("params/foo", "c:\\test")
    assert (local_dir == "c:\\test\\params\\foo") and (local_uri == "c:\\test\\params\\foo\\foo")


def test_path_to_metadata_dir_uri():
    local_dir, local_uri = path_to_metadata_dir_uri("params/foo", "c:\\test")
    assert (local_dir == "c:\\test\\params\\foo") and (local_uri == "c:\\test\\params\\foo\\foo_meta.yml")


def test_path_to_exp_run_path():
    exp_id, run_id, path = path_to_exp_run_path("18/abcdefg/params/foo")
    assert (exp_id == "18") and (run_id == "abcdefg") and (path == "params/foo")