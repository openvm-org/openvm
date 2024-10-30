import argparse
import subprocess
import os
import shutil

from metric_unify.main import generate_displayable_metrics;
from metric_unify.utils import get_git_root, create_bench_metrics_dir


def run_cargo_command(bin_name, feature_flags):
     # Get the root directory of the Git repository
    git_root = get_git_root()

    # Change the current working directory to the Git root
    os.chdir(git_root)
    create_bench_metrics_dir()
    output_path = f".bench_metrics/{bin_name}.json"
    output_path_old = None

    if os.path.exists(output_path):
        output_path_old = f"{output_path}.old"
        shutil.move(output_path, f"{output_path_old}")
        print(f"Old metrics file found, moved to {git_root}/{output_path_old}")

    # Prepare the environment variables
    env = os.environ.copy()  # Copy current environment variables
    env["OUTPUT_PATH"] = output_path
    env["RUSTFLAGS"] = "-Ctarget-cpu=native"

    # Command to run
    command = [
        "cargo", "run", "--bin", bin_name, "--release", "--features", ",".join(feature_flags)
    ]
    print(" ".join(command))

    # Run the subprocess with the updated environment
    subprocess.run(command, check=False, env=env)

    print(f"Output metrics written to {git_root}/{output_path}")

    markdown_output = generate_displayable_metrics(output_path, output_path_old)
    with open(f"{git_root}/.bench_metrics/{bin_name}.md", "w") as f:
        f.write(markdown_output)
    print(markdown_output)


def bench():
    parser = argparse.ArgumentParser()
    parser.add_argument('bench_name', type=str, help="Name of the benchmark to run")
    parser.add_argument('--features', type=str, help="Additional features")
    args = parser.parse_args()

    feature_flags = ["bench-metrics", "parallel", "mimalloc"] + ([args.features] if args.features else [])

    run_cargo_command(args.bench_name, feature_flags)


if __name__ == '__main__':
    bench()
