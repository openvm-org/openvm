#!/usr/bin/env python3
from __future__ import annotations
import shutil
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
import subprocess
import threading
from typing import Any


SDK_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = SDK_DIR / "tests"
LOG_ROOT = TESTS_DIR / "logs" / "sdk_tests"


def cargo_test_cmd(mode: str, evm=True, root=True) -> list[str]:
    assert mode in ["default", "aot", "cuda", "ignored"]
    flags = [
        "cargo",
        "nextest",
        "run",
    ]
    features = "parallel"
    if root or evm:
        features += ",root-prover"
    if evm:
        features += ",evm-verify"
    if mode == "aot":
        features += ",aot"
        flags.append("--release")
    if mode == "cuda":
        features += ",cuda"
        flags.extend(["--test-threads=1", "--cargo-profile=fast"])
    if mode == "ignored":
        flags.extend(["--run-ignored=only", "--no-tests", "pass", "--release"])
    flags.extend(["--features", features])

    return flags


PARAM_ORDER = ("app", "leaf", "internal", "root")
PARAM_ENVS = {
    "app": "APP_PARAMS_OVERRIDE",
    "leaf": "LEAF_PARAMS_OVERRIDE",
    "internal": "INTERNAL_PARAMS_OVERRIDE",
    "root": "ROOT_PARAMS_OVERRIDE",
}
DEFAULT_SWEEPS = (
    ("internal", TESTS_DIR / "internal_params.json"),
    ("root", TESTS_DIR / "root_params.json"),
)


@dataclass(frozen=True)
class ParamOverrides:
    test_cmd: list[str]
    tests_filters: list[str] = field(default_factory=lambda: [])
    app: dict[str, Any] | None = None
    leaf: dict[str, Any] | None = None
    internal: dict[str, Any] | None = None
    root: dict[str, Any] | None = None


def param_summary(overrides: ParamOverrides) -> str:
    parts = []
    for kind in PARAM_ORDER:
        params = getattr(overrides, kind)
        if params is None:
            continue

        fields = []
        for key in ("l_skip", "n_stack", "w_stack", "log_blowup"):
            if key in params:
                fields.append(f"{key}={params[key]}")

        whir = params.get("whir")
        if isinstance(whir, dict) and "k" in whir:
            fields.append(f"k_whir={whir['k']}")

        suffix = f" ({', '.join(fields)})" if fields else ""
        parts.append(f"{kind}{suffix}")

    return "; ".join(parts) if parts else "default params"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run SDK cargo tests with optional parameter override JSON files.")
    )
    parser.add_argument(
        "--log-dir", type=Path, default=None, help="dir to store the log files"
    )
    parser.add_argument("--use-evm", action="store_true")
    parser.add_argument("--use-root", action="store_true")
    parser.add_argument(
        "--test-name",
        type=str,
        help="run only these tests separated by comma",
        default=None,
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=("default", "ignored", "aot", "cuda"),
        default="default",
        help="which SDK workflow test command to run",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="maximum number of sweep configurations to run in parallel; defaults to CPU core count",
    )
    parser.add_argument(
        "--test-indices",
        type=str,
        help="optional indices of the parameters to test, comma separated, if None then test everything.",
        default=None,
    )
    parser.add_argument(
        "--sweep",
        nargs="+",
        type=Path,
        help="json files containing a list of parameters",
    )
    return parser.parse_args()


RECORD_LOCK = threading.Lock()


def make_log_dir() -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = LOG_ROOT / timestamp
    log_dir.mkdir(parents=True, exist_ok=False)
    return log_dir


def locked_append_json(path: Path, overrides: ParamOverrides):
    with RECORD_LOCK:
        if path.exists():
            with open(path, "r") as f:
                lst = json.load(f)
                assert isinstance(lst, list)
                lst.append(asdict(overrides))
            with open(path, "w") as f:
                json.dump(lst, f, indent=2)
        else:
            with open(path, "w") as f:
                json.dump([asdict(overrides)], f, indent=2)


def read_param_dict(path: Path | None):
    if path is None:
        return None
    with open(path, "r") as f:
        d = json.load(f)
    assert isinstance(d, dict)
    return d


def run_test(
    overrides: ParamOverrides, log_path: Path, successful_path: Path, failed_path: Path
):
    env = os.environ.copy()
    # env["RUST_LOG"] = "debug"
    for env_var in PARAM_ENVS.values():
        env.pop(env_var, None)
    for kind in PARAM_ORDER:
        params = getattr(overrides, kind)
        if params is not None:
            env[PARAM_ENVS[kind]] = json.dumps(params, separators=(",", ":"))

    completed = subprocess.run(
        overrides.test_cmd,
        cwd=SDK_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    output = completed.stdout
    param_record = json.dumps(asdict(overrides), indent=2)
    log_path.write_text(
        f"summary: {param_summary(overrides)}\n"
        f"parameter dict:\n{param_record}\n\n"
        f"command: {' '.join(overrides.test_cmd)}\n"
        f"cwd: {SDK_DIR}\n"
        f"exit code: {completed.returncode}\n\n"
        f"captured stdout/stderr:\n{output}"
    )

    if completed.returncode == 0:
        locked_append_json(successful_path, overrides)
    else:
        locked_append_json(failed_path, overrides)

    return completed.returncode


def get_runs(args):
    test_cmd = cargo_test_cmd(args.mode, evm=args.use_evm, root=args.use_root)

    if args.test_name is not None:
        assert isinstance(args.test_name, str)
        test_instances = args.test_name.split(",")
        assert len(test_instances) >= 1
        test_cmd.extend(["--"] + test_instances)
    else:
        test_instances = []
    jobs = args.jobs if args.jobs is not None else os.cpu_count() or 1
    if jobs < 1:
        raise ValueError("-j/--jobs must be at least 1")

    sweep_files = args.sweep

    runs = []
    test_indices = set()
    if args.test_indices is not None:
        for v in args.test_indices.split(","):
            test_indices.add(int(v))
    idx = 0
    for sweep_file in sweep_files:
        assert isinstance(sweep_file, Path)
        assert sweep_file.exists()
        with open(sweep_file, "r") as f:
            sweep = json.load(f)
        assert isinstance(sweep, list)

        for params in sweep:
            assert isinstance(params, dict)
            if len(test_indices) == 0 or idx in test_indices:
                runs.append(
                    (
                        ParamOverrides(
                            test_cmd=test_cmd,
                            tests_filters=test_instances,
                            internal=params.get("internal", None),
                            root=params.get("root", None),
                            leaf=params.get("leaf", None),
                            app=params.get("app", None),
                        )
                    )
                )
            idx += 1
    if args.num_samples is not None:
        random.seed(0)
        runs = random.sample(
            runs,
            k=min(args.num_samples, len(runs)),
        )
        print(f"randomly sampled {len(runs)} runs")
    return runs


def summarize_test(
    runs: list[ParamOverrides], return_codes: list[int], RUN_LOG_DIR
) -> Path:
    if RUN_LOG_DIR is None:
        raise RuntimeError("RUN_LOG_DIR must be initialized before writing summary")
    if len(runs) != len(return_codes):
        raise ValueError("runs and return_codes must have the same length")

    lines = ["# Successful", ""]
    for index, (run, return_code) in enumerate(zip(runs, return_codes)):
        if return_code == 0:
            lines.append(
                f"- `{index}`: {param_summary(run)} ([log](successful/{index}.log))"
            )
    if len(lines) == 2:
        lines.append("No successful parameter settings.")

    lines.extend(["", "# Failed", ""])
    failed_start = len(lines)
    for index, (run, return_code) in enumerate(zip(runs, return_codes)):
        if return_code != 0:
            lines.append(
                f"- `{index}`: {param_summary(run)} (exit code {return_code}) ([log](failed/{index}.log))"
            )
    if len(lines) == failed_start:
        lines.append("No failed parameter settings.")

    summary_path = RUN_LOG_DIR / "test_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def main() -> int:

    args = parse_args()
    runs = get_runs(args)
    RUN_LOG_DIR = make_log_dir() if args.log_dir is None else args.log_dir
    assert isinstance(RUN_LOG_DIR, Path)
    RUN_LOG_DIR.mkdir(exist_ok=True)
    log_path = RUN_LOG_DIR
    successful_logs = RUN_LOG_DIR / "successful"
    successful_logs.mkdir()
    failed_logs = RUN_LOG_DIR / "failed"
    failed_logs.mkdir()

    successful = RUN_LOG_DIR / "successful.json"
    failed = RUN_LOG_DIR / "failed.json"
    jobs = args.jobs if args.jobs is not None else os.cpu_count() or 1

    print(f"SDK directory: {SDK_DIR}")
    print(f"Logs: {RUN_LOG_DIR}")
    print(f"Running {len(runs)} parameter configuration(s)")
    print(f"Max jobs: {jobs}")

    def launch(index: int) -> int:
        overrides = runs[index]
        log_path_instance = log_path / f"{index}.log"
        returncode = run_test(overrides, log_path_instance, successful, failed)

        param_str = json.dumps(asdict(overrides))
        print("finished:")
        print(param_str)
        print(f"  with exit code {returncode}")
        print(f"  at log: {RUN_LOG_DIR}")
        if returncode != 0:
            shutil.move(log_path_instance, failed_logs / log_path_instance.name)
        else:
            shutil.move(log_path_instance, successful_logs / log_path_instance.name)
        return returncode

    return_codes = [1] * len(runs)
    if jobs == 1 or len(runs) <= 1:
        for run_index in range(len(runs)):
            print(f"[{run_index + 1}/{len(runs)}] {param_summary(runs[run_index])}")
            return_codes[run_index] = launch(run_index)
    else:
        with ThreadPoolExecutor(max_workers=min(jobs, len(runs))) as executor:
            future_to_run = {}
            for run_index, overrides in enumerate(runs):
                print(
                    f"[{run_index + 1}/{len(runs)}] queued {param_summary(overrides)}"
                )
                future = executor.submit(launch, run_index)
                future_to_run[future] = (run_index, overrides)

            for future in as_completed(future_to_run):
                run_index, overrides = future_to_run[future]
                print(f"[{run_index}/{len(runs)}] completed {param_summary(overrides)}")
                return_codes[run_index] = future.result()

    summarize_test(runs, return_codes, RUN_LOG_DIR)

    if all(return_code == 0 for return_code in return_codes):
        return 0
    else:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
