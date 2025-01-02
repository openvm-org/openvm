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
    pub diff: Option<f64>,
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
            ret.push_str(&format!("<span style='color: {}'>({})</span>", color, span));
        }
    }
    ret.push_str(&format!(" {div}"));
    ret
}

// For serialization purposes
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsFile {
    pub counter: Vec<MetricEntry>,
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
