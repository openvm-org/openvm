import json
import argparse
import os
import subprocess

from utils import FLAMEGRAPHS_DIR, create_flamegraph_dir, get_git_root, create_bench_metrics_dir


def get_stack_lines(metrics_file, stack_keys, metric_name):
    """
    Filters a metrics_file json for entries that look like:
        [ { labels: [["key1", "span1;span2"], ["key2", "span3"]], "metric": metric_name, "value": 2 } ]

    It will find entries that have all of stack_keys as present in the labels and then concatenate the corresponding values into a single flat stack entry and then add the value at the end.
    It will write a file with one line each for flamegraph.pl or inferno-flamegraph to consume.
    """
    lines = []
    with open(metrics_file, 'r') as f:
        data = json.load(f)

    # Process counters
    for counter in data.get('counter', []):
        # list of pairs -> dict
        labels = dict(counter['labels'])

        try:
            stack_values = [labels[key] for key in stack_keys]
        except KeyError:
            continue

        if counter['metric'] != metric_name:
            continue

        stack = ';'.join(stack_values)
        value = int(counter['value'])

        lines.append(f"{stack} {value}")

    # Currently cycle tracker does not use gauge
    return lines


def create_flamegraph(metrics_file, stack_keys, metric_name, reverse=False):
    lines = get_stack_lines(metrics_file, stack_keys, metric_name)

    stack_keys.remove("cycle_tracker_span")

    git_root = get_git_root()
    os.chdir(git_root)
    create_flamegraph_dir()

    fname = os.path.splitext(os.path.basename(metrics_file))[0]

    path_prefix = f"{FLAMEGRAPHS_DIR}{fname}.{'.'.join(stack_keys)}.{metric_name}{'.reverse' if reverse else ''}"
    stacks_path = f"{path_prefix}.stacks"
    flamegraph_path = f"{path_prefix}.svg"

    with open(stacks_path, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")

    with open(flamegraph_path, 'w') as f:
        command = ["inferno-flamegraph", "--title", f"{' '.join(stack_keys)} {metric_name}", stacks_path]
        if reverse:
            command.append("--reverse")

        subprocess.run(command, stdout=f, check=False)
        print (f"Created flamegraph at {flamegraph_path}")


def create_custom_flamegraphs(metrics_file):
    for reverse in [False, True]:
        create_flamegraph(metrics_file, ["cycle_tracker_span", "dsl_ir", "opcode"], "frequency", reverse=reverse)
        create_flamegraph(metrics_file, ["cycle_tracker_span", "dsl_ir", "opcode", "air_name"], "cells_used", reverse=reverse)


def main():
    import shutil

    if not shutil.which("inferno-flamegraph"):
        print("You must have inferno-flamegraph installed to use this script.")
        os.exit(1)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('metrics_json', type=str, help="Path to the metrics JSON")
    args = argparser.parse_args()

    create_custom_flamegraphs(args.metrics_json)


if __name__ == '__main__':
    main()
