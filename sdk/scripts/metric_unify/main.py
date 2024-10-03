import json
import argparse
# import yaml
# import sys

# labels is a list of (key, value) strings
def labels_to_tuple(labels):
    return tuple([tuple(pair) for pair in labels])

# Helper function to add metric data into the flat dict
def add_to_flat_dict(labels, metric, value, flat_dict):
    if labels not in flat_dict:
        flat_dict[labels] = []
    flat_dict[labels].append(Metric(metric, value))

def custom_sort_label_keys(label_key):
    """
    Custom sorting function that ensures 'group' comes first.
    Other keys are sorted alphabetically.
    """
    # Prioritize 'group' by giving it the lowest possible sort value
    if label_key == 'group':
        return (0, label_key)  # Lowest priority for 'group'
    else:
        return (1, label_key)  # Normal priority for other keys

class Metric:
    name = ""
    value = 0
    diff_value = None
    diff_percent = None

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        # Customize the string representation for printing
        diff_str = ""
        if self.diff_value is not None:
            diff_str = f", diff_value={self.diff_value}"
        if self.diff_percent is not None:
            diff_str += f", diff_percent={self.diff_percent:+.2%}"
        return f"Metric(name={self.name}, value={self.value}{diff_str})"

    def __repr__(self):
        return self.__str__()

class MetricDb:
    def __init__(self, metrics_file):
        # Dict[labels => List[Metric]]
        self.flat_dict = {}
        # Dict label_keys_tuple => Dict[label_values_tuple => List[Metric]]
        self.dict_by_label_types = {}
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        # Process counters
        for counter in data.get('counter', []):
            labels = labels_to_tuple(counter['labels'])
            metric = counter['metric']
            value = int(counter['value'])
            if value == 0:
                continue
            add_to_flat_dict(labels, metric, value, self.flat_dict)

        # Process gauges
        for gauge in data.get('gauge', []):
            labels = labels_to_tuple(gauge['labels'])
            metric = gauge['metric']
            value = float(gauge['value'])
            add_to_flat_dict(labels, metric, value, self.flat_dict)

        self.separate_by_label_types()

    def separate_by_label_types(self):
        for labels, metrics in self.flat_dict.items():
            label_keys = tuple(sorted([key for key, _ in labels], key=custom_sort_label_keys))
            label_dict = dict(labels)
            label_values = tuple([label_dict[key] for key in label_keys])
            if label_keys not in self.dict_by_label_types:
                self.dict_by_label_types[label_keys] = {}
            self.dict_by_label_types[label_keys][label_values] = metrics

# mutates db so metric dict has fields "diff_value" and "diff_percent"
def diff_metrics(db: MetricDb, db_old: MetricDb):
    for (labels, metrics) in db.flat_dict.items():
        if labels not in db_old.flat_dict:
            continue
        for metric in metrics:
            metric_old = next((m for m in db_old.flat_dict[labels] if m.name == metric.name), None)
            if metric_old:
                metric.diff_value = metric.value - metric_old.value
                if metric_old.value != 0:
                    metric.diff_percent = (metric.value - metric_old.value) / metric_old.value
    db.separate_by_label_types()

# separated_dict is dict by label types
def generate_markdown_tables(separated_dict, excluded_labels=["cycle_tracker_span"], summary_labels=["dsl_ir"]):
    markdown_output = ""

    # Loop through each set of tuple_keys
    for tuple_keys, metrics_dict in separated_dict.items():
        tuple_keys = list(tuple_keys)
        exclude = any(excluded_label in tuple_keys for excluded_label in excluded_labels)
        if exclude:
            continue

        # Check if the current tuple_keys contains any of the summary labels
        should_summarize = any(label in tuple_keys for label in summary_labels)
        if should_summarize:
            markdown_output += "<details>\n<summary>Click to expand</summary>\n\n"
        # Get all unique metric names
        metric_names = set()
        for metric_list in metrics_dict.values():
            metric_names.update([metric.name for metric in metric_list])
        metric_names = sorted(metric_names)

        # Create the table header
        header = "| " + " | ".join([f"{key}" for key in list(tuple_keys)] + metric_names) + " |"
        separator = "| " + " | ".join(["---"] * (len(tuple_keys) + len(metric_names))) + " |"
        markdown_output += header + "\n" + separator + "\n"

        # Fill the table with rows for each tuple_value and associated metrics
        for tuple_values, metrics in metrics_dict.items():
            row_values = list(tuple_values)
            row_metrics = []
            for metric_name in metric_names:
                metric = next((m for m in metrics if m.name == metric_name), None)
                metric_str = ""
                if metric:
                    metric_str += f"{metric.value:,}"
                    if metric.diff_percent is not None and metric.diff_value != 0:
                        color = "red" if metric.diff_percent > 0 else "green"
                        # Format the percentage with the color styling
                        metric_str += f' <span style="color: {color}">({metric.diff_value:+,} [{metric.diff_percent:+.1%}])</span>'
                row_metrics.append(metric_str)
            markdown_output += "| " + " | ".join(row_values + row_metrics) + " |\n"
        markdown_output += "\n"
        if should_summarize:
            markdown_output += "</details>\n\n"

    return markdown_output

# old_metrics_json is optional
def generate_displayable_metrics(metrics_json, old_metrics_json, excluded_labels=["cycle_tracker_span"], summary_labels=["dsl_ir"]):
    db = MetricDb(metrics_json)
    if old_metrics_json:
        db_old = MetricDb(old_metrics_json)
        diff_metrics(db, db_old)

    markdown_output = generate_markdown_tables(db.dict_by_label_types, excluded_labels, summary_labels)
    return markdown_output

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('metrics_json', type=str, help="Path to the metrics JSON")
    argparser.add_argument('--prev', type=str, required=False, help="Path to the previous metrics JSON for diff generation")
    argparser.add_argument('--excluded-labels', type=str, required=False, help="Comma-separated list of labels to exclude from the table")
    argparser.add_argument('--summary-labels', type=str, required=False, help="Comma-separated list of labels to include in summary rows")
    args = argparser.parse_args()

    markdown_output = generate_displayable_metrics(args.metrics_json, args.prev, args.excluded_labels.split(",") if args.excluded_labels else ["cycle_tracker_span"], args.summary_labels.split(",") if args.summary_labels else ["dsl_ir"])
    print(markdown_output)


if __name__ == '__main__':
    main()
