import argparse
import subprocess
import os
import shutil

from metric_unify.main import generate_displayable_metrics;
from metric_unify.utils import get_git_root, create_bench_metrics_dir

def run_cargo_command(
    bin_name,
    feature_flags,
    app_log_blowup,
    agg_log_blowup,
    root_log_blowup,
    internal_log_blowup,
    max_segment_length,
    instance_type,
    memory_allocator,
    output_path,
):
    # Command to run
    command = [
        "cargo", "run", "--no-default-features", "--bin", bin_name, "--profile", "maxperf", "--features", ",".join(feature_flags), "--"
    ]

    if app_log_blowup is not None:
        command.extend(["--app_log_blowup", app_log_blowup])
    if agg_log_blowup is not None:
        command.extend(["--agg_log_blowup", agg_log_blowup])
    if root_log_blowup is not None:
        command.extend(["--root_log_blowup", root_log_blowup])
    if internal_log_blowup is not None:
        command.extend(["--internal_log_blowup", internal_log_blowup])
    if max_segment_length is not None:
        command.extend(["--max_segment_length", max_segment_length])

    # Change the current working directory to the Git root
    git_root = get_git_root()
    os.chdir(git_root)
    create_bench_metrics_dir()
    output_path_old = None

    # Local only: in CI we will download the old metrics file from S3
    if os.path.exists(output_path):
        output_path_old = f"{output_path}.old"
        shutil.move(output_path, f"{output_path_old}")
        print(f"Old metrics file found, moved to {git_root}/{output_path_old}")

    # Prepare the environment variables
    env = os.environ.copy()  # Copy current environment variables
    env["OUTPUT_PATH"] = output_path
    env["RUSTFLAGS"] = "-Ctarget-cpu=native"

    # Run the subprocess with the updated environment
    subprocess.run(command, check=True, env=env)

    print(f"Output metrics written to {git_root}/{output_path}")

    # Local only: in CI the old file is not present yet and we will generate markdown in a later step.
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
    parser.add_argument('--max_segment_length', type=str, help="Max segment length for continuations")
    parser.add_argument('--features', type=str, help="Additional features")
    parser.add_argument('--output_path', type=str, required=True, help="The path to write the metrics to")
    args = parser.parse_args()

    feature_flags = ["bench-metrics", "parallel", "function-span"] + ([args.features] if args.features else []) + [args.memory_allocator]
    assert (feature_flags.count("mimalloc") + feature_flags.count("jemalloc")) == 1

    run_cargo_command(
        args.bench_name,
        feature_flags,
        args.app_log_blowup,
        args.agg_log_blowup,
        args.root_log_blowup,
        args.internal_log_blowup,
        args.max_segment_length,
        args.instance_type,
        args.memory_allocator,
        args.output_path,
    )


if __name__ == '__main__':
    bench()
