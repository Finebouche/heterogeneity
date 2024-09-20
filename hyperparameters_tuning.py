import subprocess
import json
import time
from ray import tune


def wait_for_job_and_get_result(job_name, poll_interval=60, timeout=3600):
    start_time = time.time()
    while True:
        job_status = get_job_status(job_name)

        if job_status == 'COMPLETED':
            accuracy = get_job_result(job_name)
            return accuracy
        elif job_status == 'FAILED':
            print(f"Job {job_name} failed.")
            return 0

        if time.time() - start_time > timeout:
            print(f"Job {job_name} timed out.")
            return 0

        time.sleep(poll_interval)


def get_job_status(job_name):
    cmd = ['gpulab-cli', 'status', job_name, '--json']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error getting status for job {job_name}: {stderr.decode()}")
        return 'UNKNOWN'

    status_info = json.loads(stdout.decode())
    job_status = status_info.get('status', 'UNKNOWN')
    return job_status


def get_job_result(job_name):
    cmd = ['gpulab-cli', 'download', job_name, 'result.json', '--output', f'result_{job_name}.json']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error downloading result for job {job_name}: {stderr.decode()}")
        return 0

    with open(f'result_{job_name}.json', 'r') as f:
        result_data = json.load(f)
    accuracy = result_data.get('accuracy', 0)
    return accuracy


def objective(config):
    job_name = f"ray_tune_job_{int(time.time() * 1000)}"

    # Prepare hyperparameters
    conn_add_prob = config['conn_add_prob']
    conn_delete_prob = config['conn_delete_prob']
    num_hidden = config['num_hidden']

    # Construct the command with hyperparameters
    command = (
        f"python evolve_script.py "
        f"{conn_add_prob} "
        f"{conn_delete_prob} "
        f"{num_hidden}"
    )
    # Build the job definition
    job_definition = {
        "jobDefinition": {
            "name": job_name,
            "description": "Hyperparameter tuning job submitted by Ray Tune",
            "dockerImage": "gpulab.ilabt.imec.be:5000/sample:nvidia-smi",
            "environment": {
                "CONN_ADD_PROB": str(config["conn_add_prob"]),
                "CONN_DELETE_PROB": str(config["conn_delete_prob"]),
                "NUM_HIDDEN": str(config["num_hidden"]),
                # Add other hyperparameters as needed
            },
            "command": f"cd /project_ghent/NEAT_HET && source startscript.sh && cd neat-heterogenous && {command}",
            "resources": {
                "gpus": 1,
                "systemMemory": 2000,
                "cpuCores": 2
            },
            "jobDataLocations": [
                {
                    "hostPath": "/project_ghent"
                }
            ],
            "portMappings": []
        }
    }

    cmd = ['gpulab-cli', 'submit', '--project', 'tanguy_cazalets']
    job_definition_json = json.dumps(job_definition)
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=job_definition_json.encode())

    if process.returncode != 0:
        print(f"Error submitting job {job_name}: {stderr.decode()}")
        tune.report(accuracy=0)
        return

    print(f"Successfully submitted job {job_name}")

    # Wait for the job to complete and retrieve the result
    accuracy = wait_for_job_and_get_result(job_name)

    # Report the result back to Ray Tune
    tune.report(accuracy=accuracy)


if __name__ == '__main__':
    search_space = {
        'conn_add_prob': tune.grid_search([0.1, 0.2, 0.3]),
        'conn_delete_prob': tune.grid_search([0.1, 0.2, 0.3]),
        'num_hidden': tune.choice([0, 1, 2]),
        # Add other hyperparameters as needed
    }

    tuner = tune.Tuner(
        objective,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric='accuracy',
            mode='max',
            max_concurrent_trials=5,
        ),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric='accuracy', mode='max')
    print(f"Best configuration: {best_result.config}")
    print(f"Best accuracy: {best_result.metrics['accuracy']}")
