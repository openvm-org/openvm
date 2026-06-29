#!/usr/bin/env python3
"""Extract failed log stderr sections into markdown.

Usage:
    python3 tests/failed_stderr_markdown.py tests/logs/sdk_tests/<log_dir>
    python3 tests/failed_stderr_markdown.py tests/logs/sdk_tests/<log_dir> -o failed_stderr.md

The input directory must contain a `failed/` subdirectory with `.log` files.
For each failed log, this script extracts every section after a `stderr ───`
marker and writes a markdown bullet containing the log name and the extracted
contents in a fenced code block.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


STDERR_MARKER_RE = re.compile(r"stderr\s+───", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write markdown containing stderr sections from failed logs."
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="SDK test log directory containing a failed/ subdirectory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write markdown to this file instead of stdout",
    )
    return parser.parse_args()


def extract_stderr_sections(text: str) -> str:
    matches = list(STDERR_MARKER_RE.finditer(text))
    if not matches:
        return ""

    sections = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)
    return "\n\n".join(sections)


def markdown_for_failed_logs(log_dir: Path) -> str:
    failed_dir = log_dir / "failed"
    if not failed_dir.is_dir():
        raise ValueError(f"{failed_dir} is not a directory")

    lines: list[str] = []
    for log_path in sorted(failed_dir.glob("*.log")):
        stderr = extract_stderr_sections(log_path.read_text(errors="replace"))
        if not stderr:
            stderr = "(no stderr marker found)"

        lines.append(f"- `{log_path.name}`")
        lines.append("")
        lines.append("```text")
        lines.append(stderr)
        lines.append("```")
        lines.append("")

    if not lines:
        lines.append("- No failed log files found.")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        markdown = markdown_for_failed_logs(args.log_dir)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    if args.output is None:
        print(markdown, end="")
    else:
        args.output.write_text(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
