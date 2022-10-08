import os

# +
def path_to_dir_uri(full_path, local_dir):
    local_array = [local_dir] + full_path.split("/") + [full_path.split("/")[-1]]
    local_uri = os.path.join(*local_array)
    local_dir = os.path.split(local_uri)[:-1]
    local_dir = os.path.join(*local_dir)
    return (local_dir, local_uri)

def path_to_metadata_dir_uri(full_path, local_dir):
    local_dir, local_uri = path_to_dir_uri(full_path, local_dir)
    local_uri = local_uri+"_meta.yml"
    return (local_dir, local_uri)


# -

def path_to_exp_run_path(full_path):
    path_array = full_path.split('/')
    experiment_id = path_array[0]
    run_id = path_array[1]
    path = "/".join(path_array[2:])
    return (experiment_id, run_id, path)
