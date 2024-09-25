import json
import argparse
import os
import subprocess

from utils import FLAMEGRAPHS_DIR, create_flamegraph_dir, get_git_root, create_bench_metrics_dir

# Very custom implementation right now
# Filters a metrics_file json for entries that look like:
# [ { span_label: "span1;span2;...", base_label: "opcode", "metric": metric_name, "value": 2 } ]
# It will concatenate the span_label value and base_label value into a single flat stack entry and then add the value at the end.
# It will write a file with one line each for flamegraph.pl or inferno-flamegraph to consume.
def get_stack_lines(metrics_file, base_label, metric_name, span_label="cycle_tracker_span"):
    lines = []
    with open(metrics_file, 'r') as f:
        data = json.load(f)

    # Process counters
    for counter in data.get('counter', []):
        # list of lists
        labels = counter['labels']
        span_name = None
        base_name = None
        for label in labels:
            if label[0] == span_label:
                span_name = label[1]
            if label[0] == base_label:
                base_name = label[1]
        if counter['metric'] != metric_name or span_name is None or base_name is None:
            continue
        value = int(counter['value'])
        line = f"{span_name};{base_name} {value}"
        lines.append(line)

    # Currently cycle tracker does not use gauge
    return lines

def create_flamegraph(metrics_file, base_label, metric_name, span_label="cycle_tracker_span"):
    lines = get_stack_lines(metrics_file, base_label, metric_name, span_label)

    git_root = get_git_root()
    os.chdir(git_root)
    create_flamegraph_dir()

    fname = os.path.splitext(os.path.basename(metrics_file))[0]

    path_prefix = f"{FLAMEGRAPHS_DIR}{fname}.{span_label}.{base_label}.{metric_name}"
    stacks_path = f"{path_prefix}.stacks"
    flamegraph_path = f"{path_prefix}.svg"

    with open(stacks_path, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")
    with open(flamegraph_path, 'w') as f:
        subprocess.run(["inferno-flamegraph", "--title", f"{span_label} {base_label} {metric_name}", stacks_path], stdout=f, check=False)
        print (f"Created flamegraph at {flamegraph_path}")

def create_custom_flamegraphs(metrics_file):
    create_flamegraph(metrics_file, "opcode", "frequency")
    create_flamegraph(metrics_file, "opcode", "cells_used")
    create_flamegraph(metrics_file, "chip_name", "rows_used")
    create_flamegraph(metrics_file, "dsl_ir", "frequency")
    create_flamegraph(metrics_file, "dsl_ir", "cells_used")


def main():
    print("You must have inferno-flamegraph installed to use this script.")
    argparser = argparse.ArgumentParser()
    argparser.add_argument('metrics_json', type=str, help="Path to the metrics JSON")
    args = argparser.parse_args()

    create_custom_flamegraphs(args.metrics_json)


if __name__ == '__main__':
    main()
