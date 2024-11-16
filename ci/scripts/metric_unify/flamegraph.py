import json
import argparse
import os
import subprocess

from utils import FLAMEGRAPHS_DIR, create_flamegraph_dir, get_git_root, create_bench_metrics_dir


def get_stack_lines(metrics_dict, group_by_kvs, stack_keys, metric_name):
    """
    Filters a metrics_dict obtained from json for entries that look like:
        [ { labels: [["key1", "span1;span2"], ["key2", "span3"]], "metric": metric_name, "value": 2 } ]

    It will find entries that have all of stack_keys as present in the labels and then concatenate the corresponding values into a single flat stack entry and then add the value at the end.
    It will write a file with one line each for flamegraph.pl or inferno-flamegraph to consume.
    """
    lines = []

    # Process counters
    for counter in metrics_dict.get('counter', []):
        if counter['metric'] != metric_name:
            continue
        # list of pairs -> dict
        labels = dict(counter['labels'])
        filter = False
        for key, value in group_by_kvs:
            if key not in labels or labels[key] != value:
                filter = True
                break
        if filter:
            continue

        stack_values = []
        for key in stack_keys:
            if key not in labels:
                filter = True
                break
            stack_values.append(labels[key])
        if filter:
            continue

        stack = ';'.join(stack_values)
        value = int(counter['value'])

        lines.append(f"{stack} {value}")

    # Currently cycle tracker does not use gauge
    return lines


def create_flamegraph(fname, metrics_dict, group_by_kvs, stack_keys, metric_name, reverse=False):
    lines = get_stack_lines(metrics_dict, group_by_kvs, stack_keys, metric_name)

    suffixes = [key for key in stack_keys if key != "cycle_tracker_span"]

    git_root = get_git_root()
    os.chdir(git_root)
    create_flamegraph_dir()

    path_prefix = f"{FLAMEGRAPHS_DIR}{fname}.{'.'.join(suffixes)}.{metric_name}{'.reverse' if reverse else ''}"
    stacks_path = f"{path_prefix}.stacks"
    flamegraph_path = f"{path_prefix}.svg"

    with open(stacks_path, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")

    with open(flamegraph_path, 'w') as f:
        command = ["inferno-flamegraph", "--title", f"{fname} {' '.join(suffixes)} {metric_name}", stacks_path]
        if reverse:
            command.append("--reverse")

        subprocess.run(command, stdout=f, check=False)
        print(f"Created flamegraph at {flamegraph_path}")


def create_flamegraphs(metrics_file, group_by, stack_keys, metric_name, reverse=False):
    fname_prefix = os.path.splitext(os.path.basename(metrics_file))[0]

    with open(metrics_file, 'r') as f:
        metrics_dict = json.load(f)
    # get different group_by values
    group_by_values_list = []
    for counter in metrics_dict.get('counter', []):
        labels = dict(counter['labels'])
        try:
            group_by_values_list.append(tuple([labels[group_by_key] for group_by_key in group_by]))
        except KeyError:
            continue
    # deduplicate group_by values
    group_by_values_list = list(set(group_by_values_list))
    for group_by_values in group_by_values_list:
        group_by_kvs = list(zip(group_by, group_by_values))
        fname = fname_prefix + '-' + '-'.join(group_by_values)
        create_flamegraph(fname, metrics_dict, group_by_kvs, stack_keys, metric_name, reverse=reverse)


def create_custom_flamegraphs(metrics_file, group_by=["group"]):
    for reverse in [False, True]:
        create_flamegraphs(metrics_file, group_by, ["cycle_tracker_span", "dsl_ir", "opcode"], "frequency",
                           reverse=reverse)
        create_flamegraphs(metrics_file, group_by, ["cycle_tracker_span", "dsl_ir", "opcode", "air_name"], "cells_used",
                           reverse=reverse)


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
