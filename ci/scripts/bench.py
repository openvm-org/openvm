import argparse
import os
import re
import shutil
import subprocess


def resolve_native_target_cpu(toolchain):
    target_cpus = subprocess.run(
        ["rustc", toolchain, "--print", "target-cpus"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"^\s*native.*\(currently ([^)]+)\)", target_cpus, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"failed to resolve native CPU using rustc {toolchain}")
    return match.group(1)


def run_cargo_command(
    bin_name,
    feature_flags,
    output_path,
    app_only,
    evm,
    kzg_params_dir,
    profile="release",
):
    toolchain = "+1.91"
    if "tco" in feature_flags:
        toolchain = "+nightly-2026-01-18"
    # Command to run (for best performance but slower builds, use --profile maxperf)
    command = [
        "cargo",
        toolchain,
        "run",
        "--no-default-features",
        "-p",
        "openvm-benchmarks-prove",
        "--bin",
        bin_name,
        "--profile",
        profile,
        "--features",
        ",".join(feature_flags),
        "--",
    ]

    if app_only:
        command.extend(["--app-only"])
    if evm:
        command.extend(["--evm"])
    if kzg_params_dir is not None:
        command.extend(["--kzg-params-dir", kzg_params_dir])

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
    rustflags = env.get("RUSTFLAGS", "").strip()
    if "target-cpu" not in rustflags:
        target_cpu = resolve_native_target_cpu(toolchain)
        print(f"Resolved target CPU: {target_cpu}")
        rustflags = f"{rustflags} -Ctarget-cpu={target_cpu}".strip()
    env["RUSTFLAGS"] = rustflags
    if "perf-metrics" in feature_flags:
        env["GUEST_SYMBOLS_PATH"] = os.path.splitext(output_path)[0] + ".syms"

    # Run the subprocess with the updated environment
    subprocess.run(command, check=True, env=env)

    print(f"Output metrics written to {output_path}")


def bench():
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_name", type=str, help="Name of the benchmark to run")
    parser.add_argument(
        "--kzg-params-dir",
        type=str,
        help="Directory containing KZG trusted setup files",
    )
    parser.add_argument("--features", type=str, help="Additional features")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path to write the metrics to",
    )
    parser.add_argument(
        "--app-only",
        action="store_true",
        help="Only run the app proof (skip aggregation)",
    )
    parser.add_argument(
        "--evm",
        action="store_true",
        help="Run full e2e proving (app + aggregation + root + halo2)",
    )
    args = parser.parse_args()

    feature_flags = ["metrics", "parallel"] + (
        args.features.split(",") if args.features else []
    )
    assert (feature_flags.count("mimalloc") + feature_flags.count("jemalloc")) == 1

    run_cargo_command(
        args.bench_name,
        feature_flags,
        args.output_path,
        args.app_only,
        args.evm,
        args.kzg_params_dir,
    )


if __name__ == "__main__":
    bench()
