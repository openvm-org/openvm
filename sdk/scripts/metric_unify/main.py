import json
import yaml
import sys


def label_list_to_tuple(label):
    assert len(label) == 2
    return label[0], label[1]


class MetricDb:
    metrics = {}

    def __init__(self, metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        if 'gauge' in metrics:
            self.metrics['gauge'] = {}
            for metric in metrics['gauge']:
                if metric['metric'] not in self.metrics['gauge']:
                    self.metrics['gauge'][metric['metric']] = []
                self.metrics['gauge'][metric['metric']].append(metric)

    def get_gauge_metric(self, metric_key) -> float:
        labels = set([(label['key'], label['value']) for label in metric_key['labels']])
        for metric in self.metrics['gauge'][metric_key['metric']]:
            match_tot = 0
            for label in metric['labels']:
                if label_list_to_tuple(label) in labels:
                    match_tot += 1
            if match_tot == len(labels):
                return metric['value']
        raise Exception(f"Metric not found: {metric_key}")


def read_metrics(mapping_file: str, db: MetricDb):
    with open(mapping_file, 'r') as f:
        mapping = yaml.unsafe_load(f)
    ret = {}
    for group, metrics in mapping.items():
        group_metrics = {}
        for metric in metrics:
            value = sum([db.get_gauge_metric(to_lookup) for to_lookup in metric['sum']])
            group_metrics[metric['metric']] = {
                "value": value,
                "lower_value": value,
                "upper_value": value,
            }
        ret[group] = group_metrics
    return ret


def main():
    mapping_file = sys.argv[1]
    metrics_file = sys.argv[2]
    db = MetricDb(metrics_file)
    metrics = read_metrics(mapping_file, db)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
