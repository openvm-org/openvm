use serde::{Deserialize, Serialize};

/// A file containing aggregation entries
///
/// ```rust
/// let file: AggregationFile = serde_json::from_str("{\"aggregations\": [{\"name\": \"total_time\", \"group_by\": [\"group\"], \"metrics\": [\"time\"], \"operation\": \"sum\"}]}")?;
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct AggregationFile {
    pub aggregations: Vec<AggregationEntry>,
}

/// A file containing metric entries
///
/// ```rust
/// let file: MetricsFile = serde_json::from_str("{\"counter\": [{\"labels\": [\"group\", \"bench_program_inner\"], \"name\": \"metric_name\", \"value\": 1.0}], \"gauge\": [{\"labels\": [\"group\", \"bench_program_inner\"], \"name\": \"metric_name\", \"value\": "1.0"}]}")?;
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsFile {
    pub counter: Vec<MetricEntry>,
    pub gauge: Vec<MetricEntry>,
}

/// A metric entry
///
/// ```rust
/// let metric: RowMetric = serde_json::from_str("{\"labels\": [\"group\", \"bench_program_inner\"], \"name\": \"metric_name\", \"value\": 1.0}")?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricEntry {
    #[serde(alias = "metric")]
    pub name: String,
    pub labels: Labels,
    #[serde(deserialize_with = "deserialize_f64_from_string")]
    pub value: f64,
}

/// Label identifies the type of metric
///
/// Example:
///
/// ```rust
/// let label: Label = serde_json::from_str("[\"group\", \"bench_program_innder\"]")?;
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(from = "[String; 2]")]
pub struct Label {
    pub primary: String,
    pub secondary: String,
}

impl From<[String; 2]> for Label {
    fn from([primary, secondary]: [String; 2]) -> Self {
        Self { primary, secondary }
    }
}

#[derive(Debug, Clone, Eq, Serialize, Deserialize)]
pub struct Labels(pub Vec<Label>);

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
        Labels(
            v.into_iter()
                .map(|[k, v]| Label {
                    primary: k,
                    secondary: v,
                })
                .collect(),
        )
    }
}

/// A metric entry with labels sorted
#[derive(Debug, Clone, Default)]
pub struct Metric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Sorted primary labels
    pub primary_labels: Vec<String>,
    /// Secondary labels in the same order as the primary labels
    pub secondary_labels: Vec<String>,

    // NOTE: These fields are only used for diff metrics
    /// Diff value
    pub diff: Option<f64>,
    /// Diff percentage
    pub diff_percentage: Option<f64>,
}

impl From<MetricEntry> for Metric {
    fn from(metric: MetricEntry) -> Self {
        let MetricEntry {
            name,
            mut labels,
            value,
        } = metric;

        labels.0.sort_by_key(|label| {
            if label.primary == "group" {
                (0, label.primary.clone(), label.secondary.clone()) // Prioritize 'group'
            } else {
                (1, label.primary.clone(), label.secondary.clone()) // Normal priority
            }
        });

        let primary_labels = labels.0.iter().map(|label| label.primary.clone()).collect();
        let secondary_labels = labels
            .0
            .iter()
            .map(|label| label.secondary.clone())
            .collect();

        Self {
            name,
            value,
            primary_labels,
            secondary_labels,
            ..Default::default()
        }
    }
}

impl std::hash::Hash for Metric {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.primary_labels.hash(state);
        self.secondary_labels.hash(state);
        self.name.hash(state);
    }
}

impl PartialEq for Metric {
    fn eq(&self, other: &Self) -> bool {
        self.primary_labels == other.primary_labels
            && self.name == other.name
            && self.secondary_labels == other.secondary_labels
    }
}

impl Eq for Metric {}

impl PartialOrd for Metric {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Metric {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.primary_labels
            .cmp(&other.primary_labels)
            .then(self.name.cmp(&other.name))
            .then(self.secondary_labels.cmp(&other.secondary_labels))
    }
}

/// An aggregation entry that defines how to aggregate metrics
///
/// ```rust
/// let agg: AggregationEntry = serde_json::from_str("{\"name\": \"total_time\", \"group_by\": [\"group\"], \"metrics\": [\"time\"], \"operation\": \"sum\"}")?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationEntry {
    pub name: String,
    pub group_by: Vec<String>,
    pub metrics: Vec<String>,
    pub operation: AggregationOperation,
}

/// The operation to apply to the metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationOperation {
    #[serde(rename = "sum")]
    Sum,
    #[serde(rename = "unique")]
    Unique,
}

pub fn deserialize_f64_from_string<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse::<f64>().map_err(serde::de::Error::custom)
}
