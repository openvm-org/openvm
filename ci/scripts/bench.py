import argparse
import subprocess
import os
import shutil

def run_cargo_command(
    bin_name,
    feature_flags,
    app_log_blowup,
    leaf_log_blowup,
    root_log_blowup,
    internal_log_blowup,
    max_segment_length,
    output_path,
    kzg_params_dir,
    profile="release"
):
    # Command to run (for best performance but slower builds, use --profile maxperf)
    command = [
        "cargo", "run", "--no-default-features", "-p", "openvm-benchmarks-prove", "--bin", bin_name, "--profile", profile, "--features", ",".join(feature_flags), "--"
    ]

    if app_log_blowup is not None:
        command.extend(["--app_log_blowup", app_log_blowup])
    if leaf_log_blowup is not None:
        command.extend(["--leaf_log_blowup", leaf_log_blowup])
    if root_log_blowup is not None:
        command.extend(["--root_log_blowup", root_log_blowup])
    if internal_log_blowup is not None:
        command.extend(["--internal_log_blowup", internal_log_blowup])
    if max_segment_length is not None:
        command.extend(["--max_segment_length", max_segment_length])
    if kzg_params_dir is not None:
        command.extend(["--kzg-params-dir", kzg_params_dir])
    if "profiling" in feature_flags:
        # set guest build args and vm config to profiling
        command.extend(["--profiling"])

    output_path_old = None
    # Create the output directory if it doesn't exist
    dir = os.path.dirname(output_path)
    if dir and not os.path.exists(dir):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Local only: in CI we will download the old metrics file from S3
    if os.path.exists(output_path):
        output_path_old = f"{output_path}.old"
        shutil.move(output_path, f"{output_path_old}")
        print(f"Old metrics file found, moved to {output_path_old}")

    # Prepare the environment variables
    env = os.environ.copy()  # Copy current environment variables
    env["OUTPUT_PATH"] = output_path
    if "profiling" in feature_flags:
        env["GUEST_SYMBOLS_PATH"] = os.path.splitext(output_path)[0] + ".syms"
    env["RUSTFLAGS"] = "-Ctarget-cpu=native"

    # Run the subprocess with the updated environment
    subprocess.run(command, check=True, env=env)

    print(f"Output metrics written to {output_path}")


def bench():
    parser = argparse.ArgumentParser()
    parser.add_argument('bench_name', type=str, help="Name of the benchmark to run")
    parser.add_argument('--app_log_blowup', type=str, help="Application level log blowup")
    parser.add_argument('--leaf_log_blowup', type=str, help="Leaf level log blowup")
    parser.add_argument('--root_log_blowup', type=str, help="Root level log blowup")
    parser.add_argument('--internal_log_blowup', type=str, help="Internal level log blowup")
    parser.add_argument('--max_segment_length', type=str, help="Max segment length for continuations")
    parser.add_argument('--kzg-params-dir', type=str, help="Directory containing KZG trusted setup files")
    parser.add_argument('--features', type=str, help="Additional features")
    parser.add_argument('--output_path', type=str, required=True, help="The path to write the metrics to")
    args = parser.parse_args()

    feature_flags = ["bench-metrics", "parallel"] + (args.features.split(",") if args.features else [])
    assert (feature_flags.count("mimalloc") + feature_flags.count("jemalloc")) == 1

    run_cargo_command(
        args.bench_name,
        feature_flags,
        args.app_log_blowup,
        args.leaf_log_blowup,
        args.root_log_blowup,
        args.internal_log_blowup,
        args.max_segment_length,
        args.output_path,
        args.kzg_params_dir
    )


if __name__ == '__main__':
    bench()
