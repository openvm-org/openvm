import json
import argparse
import os
import sys
import subprocess

from utils import FLAMEGRAPHS_DIR, get_git_root

def get_function_symbol(string_table, offset_str):
    try:
        offset_int = int(offset_str)
        end = string_table.find(b'\0', offset_int)
        if end == -1:
            print(f"Invalid symbol offset: {offset_int}")
            return None
        return string_table[offset_int:end].decode()
    except ValueError:
        return offset_str


def get_stack_lines(metrics_dict, group_by_kvs, stack_keys, metric_name, sum_metrics=None, string_table=None):
    """
    Filters a metrics_dict obtained from json for entries that look like:
        [ { labels: [["key1", "span1;span2"], ["key2", "span3"]], "metric": metric_name, "value": 2 } ]

    It will find entries that have all of stack_keys as present in the labels and then concatenate the corresponding values into a single flat stack entry and then add the value at the end.
    It will write a file with one line each for flamegraph.pl or inferno-flamegraph to consume.
    If sum_metrics is not None, instead of searching for metric_name, it will sum the values of the metrics in sum_metrics.
    """
    lines = []
    stack_sums = {}
    non_zero = False

    # Process counters
    for counter in metrics_dict.get('counter', []):
        if (sum_metrics is not None and counter['metric'] not in sum_metrics) or \
           (sum_metrics is None and counter['metric'] != metric_name):
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
            if key == 'cycle_tracker_span':
                if labels[key] == '' or string_table is None:
                    stack_values.append(labels[key])
                else:
                    symbol_offsets = labels[key].split(';')
                    function_symbols = [get_function_symbol(string_table, offset) for offset in symbol_offsets]
                    stack_values.extend(function_symbols)
            else:
                # don't make a stack frame for empty label
                if labels[key] == '':
                    continue
                stack_values.append(labels[key])
        if filter:
            continue

        stack = ';'.join(stack_values)
        value = int(counter['value'])
        stack_sums[stack] = stack_sums.get(stack, 0) + value

        if value != 0:
            non_zero = True

    lines = [f"{stack} {value}" for stack, value in stack_sums.items() if value != 0]

    # Currently cycle tracker does not use gauge
    return lines if non_zero else []


def create_flamegraph(fname, metrics_dict, group_by_kvs, stack_keys, metric_name, sum_metrics=None, reverse=False, string_table=None):
    lines = get_stack_lines(metrics_dict, group_by_kvs, stack_keys, metric_name, sum_metrics, string_table)
    if not lines:
        return

    suffixes = [key for key in stack_keys if key != "cycle_tracker_span"]

    git_root = get_git_root()
    flamegraph_dir = os.path.join(git_root, FLAMEGRAPHS_DIR)
    os.makedirs(flamegraph_dir, exist_ok=True)

    path_prefix = f"{flamegraph_dir}{fname}.{'.'.join(suffixes)}.{metric_name}{'.reverse' if reverse else ''}"
    stacks_path = f"{path_prefix}.stacks"
    flamegraph_path = f"{path_prefix}.svg"

    with open(stacks_path, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")

    with open(flamegraph_path, 'w') as f:
        command = ["inferno-flamegraph", "--title", f"{fname} {' '.join(suffixes)} {metric_name}", stacks_path]
        if reverse:
            command.append("--reverse")
            command.append("--inverted")

        subprocess.run(command, stdout=f, check=False)
        print(f"Created flamegraph at {flamegraph_path}")


def create_flamegraphs(metrics_file, group_by, stack_keys, metric_name, sum_metrics=None, reverse=False, string_table=None):
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
        create_flamegraph(fname, metrics_dict, group_by_kvs, stack_keys, metric_name, sum_metrics, reverse=reverse, string_table=string_table)


def create_custom_flamegraphs(metrics_file, group_by=["group"], string_table=None):
    for reverse in [False, True]:
        create_flamegraphs(metrics_file, group_by, ["cycle_tracker_span", "dsl_ir", "opcode"], "frequency",
                           reverse=reverse, string_table=string_table)
        create_flamegraphs(metrics_file, group_by, ["cycle_tracker_span", "dsl_ir", "opcode", "air_name"], "cells_used",
                           reverse=reverse, string_table=string_table)
        create_flamegraphs(metrics_file, group_by, ["cell_tracker_span"], "cells_used",
                           sum_metrics=["simple_advice_cells", "fixed_cells", "lookup_advice_cells"],
                           reverse=reverse, string_table=string_table)


def main():
    import shutil

    if not shutil.which("inferno-flamegraph"):
        print("You must have inferno-flamegraph installed to use this script.")
        sys.exit(1)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('metrics_json', type=str, help="Path to the metrics JSON")
    argparser.add_argument('--guest-symbols', type=str, help="Path to the guest symbols file", default=None, required=False)
    args = argparser.parse_args()

    if args.guest_symbols:
        with open(args.guest_symbols, 'rb') as f:
            string_table = f.read()
    else:
        string_table = None

    create_custom_flamegraphs(args.metrics_json, string_table=string_table)


if __name__ == '__main__':
    main()
