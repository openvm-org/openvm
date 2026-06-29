#!/usr/bin/env python3
"""Generate a CI markdown summary for SDK sweep logs.

Usage:
    python3 tests/post_test_summarize.py tests/logs/sdk_tests/<log_dir>
    python3 tests/post_test_summarize.py tests/logs/sdk_tests/<log_dir1> \
        tests/logs/sdk_tests/<log_dir2> -o tests/logs/sdk_tests/ci_summary.md
    python3 tests/post_test_summarize.py tests/logs/sdk_tests/<log_dir> \
        -o tests/logs/sdk_tests/<log_dir>/ci_summary.md

Each log directory must contain `successful/` and `failed/` log subdirectories
created by `tests/run_sdk_tests2.py`. The output concatenates summaries for all
log directories, containing the successful and failed parameter settings plus a
concise stderr excerpt for each failed log. Log references are artifact-internal
paths; the workflow adds the artifact URL.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from urllib.parse import quote


CAPTURED_OUTPUT_MARKER = "captured stdout/stderr:\n"
STDERR_MARKER_RE = re.compile(r"stderr\s+───", re.IGNORECASE)
SUMMARY_RE = re.compile(r"^summary:\s*(.*)$", re.MULTILINE)
EXIT_CODE_RE = re.compile(r"^exit code:\s*(-?\d+)\s*$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CI markdown for SDK parameter sweep log directories."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write markdown to this file; defaults to <first_log_dir>/ci_summary.md",
    )
    parser.add_argument(
        "--artifact-path-prefix",
        type=str,
        default="",
        help=(
            "prefix to prepend to log paths, for example "
            "crates/sdk/tests/logs/sdk_tests/<timestamp>"
        ),
    )
    parser.add_argument(
        "--max-stderr-lines",
        type=int,
        default=60,
        help="maximum stderr lines to include per failed log",
    )
    parser.add_argument(
        "--log-dirs",
        nargs="+",
        type=Path,
        help="SDK sweep log directories containing successful/ and failed/ logs",
    )
    return parser.parse_args()


def log_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.stem), path.name)
    except ValueError:
        return (10**12, path.name)


def read_text(path: Path) -> str:
    return path.read_text(errors="replace")


def read_log_summary(log_text: str) -> str:
    match = SUMMARY_RE.search(log_text)
    return match.group(1).strip() if match else "unknown parameter setting"


def read_exit_code(log_text: str) -> str:
    match = EXIT_CODE_RE.search(log_text)
    return match.group(1) if match else "unknown"


def truncate_lines(text: str, max_lines: int) -> str:
    if max_lines <= 0:
        return text
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines] + ["... (truncated)"])


def truncate_tail_lines(text: str, max_lines: int) -> str:
    if max_lines <= 0:
        return text
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(["... (truncated)"] + lines[-max_lines:])


def extract_stderr(log_text: str, max_lines: int) -> str:
    matches = list(STDERR_MARKER_RE.finditer(log_text))
    if not matches:
        _, marker, output = log_text.partition(CAPTURED_OUTPUT_MARKER)
        if marker:
            return truncate_tail_lines(output.strip(), max_lines)
        return "(no captured output found)"

    sections: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(log_text)
        section = log_text[start:end].strip()
        if section:
            sections.append(section)

    stderr = "\n\n".join(sections) if sections else "(empty stderr section)"
    return truncate_lines(stderr, max_lines)


def log_ref(status: str, log_path: Path, artifact_path_prefix: str) -> str:
    if artifact_path_prefix:
        target = f"{artifact_path_prefix.rstrip('/')}/{status}/{log_path.name}"
    else:
        target = f"{status}/{log_path.name}"
    return f"[log]({quote(target, safe='/')})"


def artifact_path_prefix_for(
    log_dir: Path, artifact_path_prefix: str, multiple_log_dirs: bool
) -> str:
    if not multiple_log_dirs:
        return artifact_path_prefix
    if artifact_path_prefix:
        return f"{artifact_path_prefix.rstrip('/')}/{log_dir.name}"
    return log_dir.name


def markdown_heading(level: int, title: str) -> str:
    return f"{'#' * level} {title}"


def append_log_list(
    lines: list[str],
    title: str,
    status: str,
    logs: list[Path],
    artifact_path_prefix: str,
    heading_level: int,
) -> None:
    lines.extend([markdown_heading(heading_level, title), ""])
    if not logs:
        lines.append(f"No {title.lower()} parameter settings.")
        lines.append("")
        return

    for log_path in logs:
        log_text = read_text(log_path)
        run_id = log_path.stem
        summary = read_log_summary(log_text)
        log = log_ref(status, log_path, artifact_path_prefix)
        if status == "failed":
            exit_code = read_exit_code(log_text)
            lines.append(f"- `{run_id}`: {summary} (exit code {exit_code}) ({log})")
        else:
            lines.append(f"- `{run_id}`: {summary} ({log})")
    lines.append("")


def generate_summary(
    log_dir: Path,
    artifact_path_prefix: str,
    max_stderr_lines: int,
    title: str | None = None,
) -> str:
    successful_logs = sorted((log_dir / "successful").glob("*.log"), key=log_sort_key)
    failed_logs = sorted((log_dir / "failed").glob("*.log"), key=log_sort_key)

    lines: list[str] = []
    heading_level = 1
    if title is not None:
        lines.extend([markdown_heading(heading_level, title), ""])
        heading_level += 1
    append_log_list(
        lines,
        "Successful",
        "successful",
        successful_logs,
        artifact_path_prefix,
        heading_level,
    )
    append_log_list(
        lines, "Failed", "failed", failed_logs, artifact_path_prefix, heading_level
    )

    lines.extend([markdown_heading(heading_level, "Failed stderr"), ""])
    if not failed_logs:
        lines.append("No failed log files found.")
        lines.append("")
    else:
        for log_path in failed_logs:
            log_text = read_text(log_path)
            log = log_ref("failed", log_path, artifact_path_prefix)
            lines.append(f"- `{log_path.name}` ({log})")
            lines.append("")
            lines.append("```text")
            lines.append(extract_stderr(log_text, max_stderr_lines))
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def generate_summaries(
    log_dirs: list[Path], artifact_path_prefix: str, max_stderr_lines: int
) -> str:
    multiple_log_dirs = len(log_dirs) > 1
    summaries = []
    for log_dir in log_dirs:
        summary_title = f"Log directory: `{log_dir}`" if multiple_log_dirs else None
        summaries.append(
            generate_summary(
                log_dir,
                artifact_path_prefix_for(
                    log_dir, artifact_path_prefix, multiple_log_dirs
                ),
                max_stderr_lines,
                summary_title,
            )
        )
    return "\n".join(summaries)


def main() -> int:
    args = parse_args()
    for log_dir in args.log_dirs:
        if not log_dir.is_dir():
            raise SystemExit(f"error: {log_dir} is not a directory")

    output = args.output or args.log_dirs[0] / "ci_summary.md"
    output.write_text(
        generate_summaries(
            args.log_dirs, args.artifact_path_prefix, args.max_stderr_lines
        )
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
