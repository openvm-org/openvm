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
        flat_dict[labels] = {}
    flat_dict[labels][metric] = value

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


class MetricDb:
    # Dict labels => Dict[metric_name, value]
    flat_dict = {}
    # Dict label_keys_tuple => Dict[label_values_tuple => Dict[metric_name, value]]
    dict_by_label_types = {}

    def __init__(self, metrics_file):
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        # Process counters
        for counter in data.get('counter', []):
            labels = labels_to_tuple(counter['labels'])
            metric = counter['metric']
            value = int(counter['value'])
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


# separated_dict is dict by label types
def generate_markdown_tables(separated_dict):
    markdown_output = ""

    # Loop through each set of tuple_keys
    for tuple_keys, metrics_dict in separated_dict.items():
        # Get all unique metric names
        metric_names = set()
        for values_dict in metrics_dict.values():
            metric_names.update(values_dict.keys())
        metric_names = sorted(metric_names)

        # Create the table header
        header = "| " + " | ".join([f"{key}" for key in list(tuple_keys)] + metric_names) + " |"
        separator = "| " + " | ".join(["---"] * (len(tuple_keys) + len(metric_names))) + " |"
        markdown_output += header + "\n" + separator + "\n"

        # Fill the table with rows for each tuple_value and associated metrics
        for tuple_values, metrics in metrics_dict.items():
            row_values = list(tuple_values)
            row_metrics = [str(metrics.get(metric, "")) for metric in metric_names]  # Fill missing metrics with empty string
            markdown_output += "| " + " | ".join(row_values + row_metrics) + " |\n"
        markdown_output += "\n"

    return markdown_output


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('metrics_json', type=str, help="Path to the metrics JSON")
    args = argparser.parse_args()

    db = MetricDb(args.metrics_json)

    markdown_output = generate_markdown_tables(db.dict_by_label_types)
    print(markdown_output)


if __name__ == '__main__':
    main()
