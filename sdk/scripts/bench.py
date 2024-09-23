import argparse
import subprocess
import os

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
    folder_path = ".bench_metrics/"

    # Use os.makedirs to create the folder if it doesn't exist
    # The exist_ok=True argument ensures no error is raised if the directory already exists
    os.makedirs(folder_path, exist_ok=True)

def run_cargo_command(bin_name, feature_flags):
     # Get the root directory of the Git repository
    git_root = get_git_root()

    # Change the current working directory to the Git root
    os.chdir(git_root)
    create_bench_metrics_dir()
    output_path = f".bench_metrics/{bin_name}.json"

    # Prepare the environment variables
    env = os.environ.copy()  # Copy current environment variables
    env["OUTPUT_PATH"] = output_path
    env["RUSTFLAGS"] = "-Ctarget-cpu=native"

    # Command to run
    command = [
        "cargo", "run", "--bin", bin_name, "--release", "--features", feature_flags
    ]

    try:
        # Run the subprocess with the updated environment
        subprocess.run(command, check=True, env=env)

    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with error: {e}")

    print(f"Output metrics written to {git_root}/{output_path}")

def bench():
    parser = argparse.ArgumentParser()
    parser.add_argument('bench_name', type=str, help="Name of the benchmark to run")
    parser.add_argument('--features', type=str, help="Additional features")
    args = parser.parse_args()

    feature_flags = "bench-metrics,parallel,mimalloc" + ("," + args.features if args.features else "")

    run_cargo_command(args.bench_name, feature_flags)

if __name__ == '__main__':
    bench()
