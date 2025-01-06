import subprocess
import os

BENCH_METRICS_DIR = ".bench_metrics/"
FLAMEGRAPHS_DIR = ".bench_metrics/flamegraphs/"

def get_git_root():
    # Run the git command to get the root directory
    result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if the command was successful
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        raise RuntimeError("Failed to find the root of the Git repository. Are you in a Git repo?")

def create_bench_metrics_dir():
    # Define the path to the folder
    folder_path = BENCH_METRICS_DIR

    # Use os.makedirs to create the folder if it doesn't exist
    # The exist_ok=True argument ensures no error is raised if the directory already exists
    os.makedirs(folder_path, exist_ok=True)
