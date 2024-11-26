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


def objective(cpus_per_job=4, num_generations=100, sweep_id=None, project=None):
    job_name = f"wandb_{int(time.time() * 1000)}"

    # The file to call
    command = (
        f"/project_ghent/NEAT_HET/neat_het_env/bin/python server_tuning_script.py "
    )
    # Build the job definition
    job_definition = {
        "name": job_name,
        "owner": {
            "projectUrn": "urn:publicid:IDN+ilabt.imec.be+project+tanguy_cazalets"
        },
        "request": {
            "docker": {
                "image": "tcazalet:PrBwbmivszbQgHnGNCqt@gitlab.ilabt.imec.be:4567/tcazalet/docker_container:mujoco",
                "command": f"sh -c \"cd /project_ghent/NEAT_HET/neat-heterogeneous && "
                           f"{command}\"",
                "environment": {
                    "SWEEP_ID": sweep_id,
                    "WANDB_PROJECT": project,
                    "CPUS_PER_JOB": cpus_per_job,
                    "NUM_GENERATIONS": num_generations,
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
                "cpus": cpus_per_job,
                "cpuMemoryGb": 5
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

    # Submit the job
    job_id = submit_job(job_definition)


if __name__ == '__main__':
    problem = "gym"  # mnist or gym
    project = f"neat-{problem}"

    # Initialize the sweep
    for num_hidden in [5]:
        sweep_configuration = {
            "name": f"sweep-{problem}-{num_hidden}",
            "method": "bayes",
            "metric": {"goal": "maximize", "name": "val_score"},
            "parameters": {
                'conn_add_prob': {"min": 0.1, "max": 0.9},
                'conn_delete_prob': {"min": 0.1, "max": 0.9},
                'activation_options': {"values":  [
                    'tanh',
                    "sigmoid tanh sin gauss relu softplus identity clamped abs hat"
                ]},
                "initial_connection": {"values": ["partial_direct 0.1", "partial_direct 0.25", "partial_direct 0.5"]},
                'num_hidden': {'values': [num_hidden]},
                'activation_mutate_rate': {"min": 0.1, "max": 0.9},
                'weight_mutate_rate': {"min": 0.1, "max": 0.9},
                'bias_mutate_rate': {"min": 0.1, "max": 0.9},
                'enabled_mutate_rate': {"min": 0.1, "max": 0.9},
                'species_elitism': {'min': 1, 'max': 5},
            },
        }

        sweep_id = None
        if sweep_id is None:
            # Create a new sweep
            print("Creating new sweep...")
            sweep_id = wandb.sweep(sweep=sweep_configuration, project=project, entity="tcazalet_airo")
        print(f"Sweep ID: {sweep_id}")

        # Generate all combinations of hyperparameters
        # keys, values = zip(*search_space.items())
        # experiments = [dict(zip(keys, v)) for v in product(*values)]
        # for config in experiments:`
        # start 10 jobs
        for _ in range(50):
            objective(cpus_per_job=6, num_generations=500, sweep_id=sweep_id, project=f"neat-{problem}")

