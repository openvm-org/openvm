use std::collections::{BTreeMap, HashMap};

use num_format::{Locale, ToFormattedString};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub value: f64,
}

impl Metric {
    pub fn new(name: String, value: f64) -> Self {
        Self { name, value }
    }
}

#[derive(Debug, Clone, Eq)]
pub struct Labels(pub Vec<(String, String)>);

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct MdTableCell {
    pub val: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diff: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BencherValue {
    pub value: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lower_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upper_value: Option<f64>,
}

/// Benchmark output in [Bencher Metric Format](https://bencher.dev/docs/reference/bencher-metric-format/).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BenchmarkOutput {
    // BMF max depth is 2
    #[serde(flatten)]
    pub by_name: HashMap<String, HashMap<String, BencherValue>>,
}

impl Labels {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0
            .iter()
            .find_map(|(k, v)| (k == key).then_some(v.as_str()))
    }

    pub fn remove(&mut self, key: &str) {
        self.0.retain(|(k, _)| k != key);
    }
}

impl PartialEq for Labels {
    fn eq(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }
        let mut self_sorted = self.0.clone();
        let mut other_sorted = other.0.clone();
        self_sorted.sort();
        other_sorted.sort();
        self_sorted == other_sorted
    }
}

impl std::hash::Hash for Labels {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let mut sorted = self.0.clone();
        sorted.sort();
        sorted.hash(state);
    }
}

impl From<Vec<[String; 2]>> for Labels {
    fn from(v: Vec<[String; 2]>) -> Self {
        Labels(v.into_iter().map(|[k, v]| (k, v)).collect())
    }
}

#[derive(Debug, Default)]
pub struct MetricDb {
    pub flat_dict: HashMap<Labels, Vec<Metric>>,
    pub dict_by_label_types: HashMap<Vec<String>, BTreeMap<Vec<String>, Vec<Metric>>>,
}

impl MetricDb {
    pub fn format_number(value: f64) -> String {
        let whole = value.trunc() as i64;
        let decimal = (value.fract() * 100.0).abs().round() as i64;

        if decimal == 0 {
            whole.to_formatted_string(&Locale::en)
        } else {
            format!("{}.{:02}", whole.to_formatted_string(&Locale::en), decimal)
        }
    }
}

impl MdTableCell {
    pub fn new(val: f64, diff: Option<f64>) -> Self {
        Self { val, diff }
    }
}

impl std::fmt::Display for MdTableCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let div = MetricDb::format_number(self.val);
        if let Some(diff) = self.diff {
            let color = if diff > 0.0 { "red" } else { "green" };
            let original_val = self.val - diff;
            let diff_percent = diff / original_val;
            let span = format!("{:+.0} [{:+.1}%]", diff, diff_percent * 100.0);
            if diff_percent.abs() < 0.001 {
                write!(f, "{}", format_cell(&div, None, None))
            } else {
                write!(f, "{}", format_cell(&div, Some(&span), Some(color)))
            }
        } else {
            write!(f, "{}", format_cell(&div, None, None))
        }
    }
}
fn format_cell(div: &str, span: Option<&str>, span_color: Option<&str>) -> String {
    let mut ret = String::new();
    if let Some(span) = span {
        if let Some(color) = span_color {
            ret.push_str(&format!("<span style='color: {color}'>({span})</span>"));
        }
    }
    ret.push_str(&format!(" {div}"));
    ret
}

impl BencherValue {
    pub fn new(value: f64) -> Self {
        Self {
            value,
            lower_value: None,
            upper_value: None,
        }
    }
}

impl From<MdTableCell> for BencherValue {
    fn from(cell: MdTableCell) -> Self {
        Self::new(cell.val)
    }
}

// For serialization purposes
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsFile {
    #[serde(default)]
    pub counter: Vec<MetricEntry>,
    #[serde(default)]
    pub gauge: Vec<MetricEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricEntry {
    pub labels: Vec<[String; 2]>,
    pub metric: String,
    #[serde(deserialize_with = "deserialize_f64_from_string")]
    pub value: f64,
}

pub fn deserialize_f64_from_string<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse::<f64>().map_err(serde::de::Error::custom)
}
