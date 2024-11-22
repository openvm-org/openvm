import argparse
import subprocess
import os
import shutil

from metric_unify.main import generate_displayable_metrics;
from metric_unify.utils import get_git_root, create_bench_metrics_dir


def run_cargo_command(bin_name, feature_flags, app_log_blowup, agg_log_blowup, root_log_blowup, internal_log_blowup, instance_type, memory_allocator):
    # Command to run
    command = [
        "cargo", "run", "--no-default-features", "--bin", bin_name, "--profile", "maxperf", "--features", ",".join(feature_flags), "--"
    ]
    output_path = f".bench_metrics/{bin_name}"

    if app_log_blowup is not None:
        command.extend(["--app_log_blowup", app_log_blowup])
        output_path += f"-{app_log_blowup}"
    if agg_log_blowup is not None:
        command.extend(["--agg_log_blowup", agg_log_blowup])
        output_path += f"-{agg_log_blowup}"
    if root_log_blowup is not None:
        command.extend(["--root_log_blowup", root_log_blowup])
        output_path += f"-{root_log_blowup}"
    if internal_log_blowup is not None:
        command.extend(["--internal_log_blowup", internal_log_blowup])
        output_path += f"-{internal_log_blowup}"
    output_path += f"-{instance_type}-{memory_allocator}.json"

    # Change the current working directory to the Git root
    git_root = get_git_root()
    os.chdir(git_root)
    create_bench_metrics_dir()
    output_path_old = None

    if os.path.exists(output_path):
        output_path_old = f"{output_path}.old"
        shutil.move(output_path, f"{output_path_old}")
        print(f"Old metrics file found, moved to {git_root}/{output_path_old}")

    # Prepare the environment variables
    env = os.environ.copy()  # Copy current environment variables
    env["OUTPUT_PATH"] = output_path
    env["RUSTFLAGS"] = "-Ctarget-cpu=native"

    # Run the subprocess with the updated environment
    subprocess.run(command, env=env)

    print(f"Output metrics written to {git_root}/{output_path}")

    markdown_output = generate_displayable_metrics(output_path, output_path_old)
    with open(f"{git_root}/.bench_metrics/{bin_name}.md", "w") as f:
        f.write(markdown_output)


def bench():
    parser = argparse.ArgumentParser()
    parser.add_argument('bench_name', type=str, help="Name of the benchmark to run")
    parser.add_argument('--instance_type', type=str, required=True, help="Instance this benchmark is running on")
    parser.add_argument('--memory_allocator', type=str, required=True, help="Memory allocator for this benchmark")
    parser.add_argument('--app_log_blowup', type=str, help="Application level log blowup")
    parser.add_argument('--agg_log_blowup', type=str, help="Aggregation level log blowup")
    parser.add_argument('--root_log_blowup', type=str, help="Application level log blowup")
    parser.add_argument('--internal_log_blowup', type=str, help="Aggregation level log blowup")
    parser.add_argument('--features', type=str, help="Additional features")
    args = parser.parse_args()

    feature_flags = ["bench-metrics", "parallel", "function-span"] + ([args.features] if args.features else []) + [args.memory_allocator]
    assert (feature_flags.count("mimalloc") + feature_flags.count("jemalloc")) == 1

    run_cargo_command(args.bench_name, feature_flags, args.app_log_blowup, args.agg_log_blowup, args.root_log_blowup, args.internal_log_blowup, args.instance_type, args.memory_allocator)


if __name__ == '__main__':
    bench()
