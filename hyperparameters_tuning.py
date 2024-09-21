import subprocess
import json
import time
import tempfile
import wandb


def submit_job(job_definition):
    # Create a temporary file for the job definition
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(job_definition, temp_file, indent=4)
        temp_file_name = temp_file.name  # Store the name of the temporary file

    # Submit the job using gpulab-cli
    cmd = ['gpulab-cli', 'submit', '--project', 'tanguy_cazalets', temp_file_name]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error submitting job: {stderr.decode()}")
        return None
    else:
        # Extract job ID from the response
        print(f"Job submitted successfully: {stdout.decode()}")
        job_id = extract_job_id(stdout.decode())  # Implement extract_job_id
        print(job_id)
        return job_id


def extract_job_id(submit_output):
    # Extract the job ID from the submission output
    # Assuming the output contains the job ID in a known format
    # Example logic based on actual gpulab-cli output
    return submit_output.strip().split()[-1]  # Placeholder logic


def wait_for_job(job_id):
    # Use gpulab-cli wait to wait for job completion
    cmd = ['gpulab-cli', 'wait', job_id, '--wait-done']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error waiting for job {job_id}: {stderr.decode()}")
        return False
    else:
        print(f"Job {job_id} completed successfully")
        return True


def objective(config, sweep_id=None):
    job_name = f"ray_{int(time.time() * 1000)}"

    # Construct the command with hyperparameters
    command = (
        f"/project_ghent/NEAT_HET/neat_het_env/bin/python server_tuning_script.py "
        # f"{config['conn_add_prob']} "
        # f"{config['conn_delete_prob']} "
        # f"{config['num_hidden']} "
        # f"{config['activation_options']} "
    )
    # Build the job definition
    job_definition = {
        "name": job_name,
        "owner": {
            "projectUrn": "urn:publicid:IDN+ilabt.imec.be+project+tanguy_cazalets"
        },
        "request": {
            "docker": {
                "image": "jupyter/tensorflow-notebook",
                "command": f"sh -c \"cd /project_ghent/NEAT_HET/neat-heterogeneous && "
                           f"{command}\"",
                "environment": {
                    "CONN_ADD_PROB": str(config['conn_add_prob']),
                    "CONN_DELETE_PROB": str(config['conn_delete_prob']),
                    "NUM_HIDDEN": str(config['num_hidden']),
                    "ACTIVATION_OPTIONS": f"{config['activation_options']}",
                    "SWEEP_ID": sweep_id,
                },
                "storage": [
                    {
                        "hostPath": "/project_ghent",
                        "containerPath": "/project_ghent"
                    }
                ]
            },
            "resources": {
                "gpus": 0,
                "cpus": 4,
                "cpuMemoryGb": 10
            },
            "scheduling": {
                "interactive": False,
                "restartable": False,
                "minDuration": "14 days",
                "maxDuration": "14 days",
                "reservationIds": []
            },
            "extra": {
                "sshPubKeys": [
                    "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDXdEvo2XUdIf5YbWY0jrIcdxLWb9vbMmBkiEZlKvK8iSG9eJkYAYyVDjEXAQQz/eCW1hpNAiQ5eI2y4xAfOonCZqHb23OsTu2u2fsguKnFwDNBIjz3qGXLK2D3OIHpeZsTlxPjd+tm6Blp+YWpXg6YW+UyqAYPV08Ff53VUHS9MEL318TY7rdSvlKeoed6VfBsuNmxaatCxQuZDz08VzYYl3wLW1TZH5lmulzUcKTzE8F60gK+F3M2EDwdHabFAnxQgcuZCiNSJrvFJyeVHpMnFLwfWawdaS8TO3VBozRv5YszWyNTnW/BK6XIP+soqc9z5BJgkiHl4BSxDKCJ8piP"
                ]
            }
        }
    }

    print(job_definition['request']['docker']['environment'])

    # Submit the job
    job_id = submit_job(job_definition)


if __name__ == '__main__':
    from itertools import product

    search_space = {
        'conn_add_prob': [0.1],
        'conn_delete_prob': [0.1],
        'num_hidden': [2],
        'activation_options': [
            'tanh',
            "sigmoid tanh sin gauss relu softplus identity clamped abs hat"
        ]
    }

    # Generate all combinations of hyperparameters
    keys, values = zip(*search_space.items())
    experiments = [dict(zip(keys, v)) for v in product(*values)]

    # Optional: Initialize a sweep configuration in wandb
    sweep_configuration = {
        "name": "my-first-sweep",
        "method": "grid",
        "parameters": {
            "conn_add_prob": {"values": search_space['conn_add_prob']},
            "conn_delete_prob": {"values": search_space['conn_delete_prob']},
            "num_hidden": {"values": search_space['num_hidden']},
            "activation_options": {"values": search_space['activation_options']},
        },
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

    for config in experiments:
        objective(config, sweep_id)

