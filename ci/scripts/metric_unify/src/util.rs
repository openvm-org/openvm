use std::collections::HashSet;

use itertools::Itertools;

use crate::{
    markdown::{format_metric_value, Table, TableHeader, TableRow},
    metric::Metric,
};

/// A group of metrics with the same primary labels
pub struct GroupedMetricsByPrimaryLabels {
    metrics: Vec<Metric>,
}

impl GroupedMetricsByPrimaryLabels {
    pub fn new(metrics: Vec<Metric>) -> Self {
        debug_assert!(!metrics.is_empty());
        debug_assert!(metrics
            .iter()
            .all(|m| m.primary_labels == metrics[0].primary_labels));
        debug_assert!(metrics.iter().all_unique());
        Self { metrics }
    }

    pub fn primary_labels(&self) -> Vec<String> {
        self.metrics[0].primary_labels.clone()
    }

    pub fn metric_names(&self) -> Vec<String> {
        let mut names = self
            .metrics
            .iter()
            .map(|m| m.name.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        names.sort();
        names
    }

    // Table header: primary labels + metric names
    // Table rows: secondary labels + metric values
    #[allow(clippy::wrong_self_convention)]
    pub fn to_table(self) -> Table {
        let primary_labels = self.primary_labels();
        let metric_names = self.metric_names();
        let table_header = TableHeader::new([primary_labels, metric_names.clone()].concat());

        let mut metrics = self.metrics;
        metrics.sort_by_key(|m| m.secondary_labels.clone());
        let table_rows = metrics
            .into_iter()
            .chunk_by(|m| m.secondary_labels.clone())
            .into_iter()
            .map(|(_, group)| {
                GroupedMetricsBySecondaryLabels::new(group.collect()).to_row(&metric_names)
            })
            .collect();

        Table::new(table_header, table_rows)
    }
}

pub struct GroupedMetricsBySecondaryLabels {
    metrics: Vec<Metric>,
}

impl GroupedMetricsBySecondaryLabels {
    pub fn new(metrics: Vec<Metric>) -> Self {
        debug_assert!(!metrics.is_empty());
        debug_assert!(metrics
            .iter()
            .all(|m| m.secondary_labels == metrics[0].secondary_labels));
        debug_assert!(metrics.iter().all_unique());
        Self { metrics }
    }

    pub fn secondary_labels(&self) -> Vec<String> {
        self.metrics[0].secondary_labels.clone()
    }

    pub fn to_row(&self, metric_names: &[String]) -> TableRow {
        let secondary_labels = self.secondary_labels();
        let mut row_values = Vec::new();
        for name in metric_names.iter() {
            let value_str = self
                .metrics
                .iter()
                .find(|m| &m.name == name)
                .map(format_metric_value)
                .unwrap_or(String::default());
            row_values.push(value_str);
        }
        TableRow::new(secondary_labels, row_values)
    }
}

pub fn to_tables(metrics: Vec<Metric>) -> Vec<Table> {
    // Remove duplicate metrics
    let mut metrics = metrics
        .into_iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    // Sort metrics by primary labels for later grouping
    metrics.sort_by_key(|m| m.primary_labels.clone());

    metrics
        .into_iter()
        .chunk_by(|m| m.primary_labels.clone())
        .into_iter()
        .map(|(_, group)| group.collect())
        .map(GroupedMetricsByPrimaryLabels::new)
        .map(GroupedMetricsByPrimaryLabels::to_table)
        .collect()
}
