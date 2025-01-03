use std::fmt::Display;

use num_format::{Locale, ToFormattedString};

use crate::metric::Metric;

const TABLE_SEPARATOR: &str = "|";
const COLUMN_SEPARATOR: &str = "---";

pub struct Tables {
    tables: Vec<Table>,
}

impl From<Vec<Table>> for Tables {
    fn from(tables: Vec<Table>) -> Self {
        Self { tables }
    }
}

impl Display for Tables {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.tables
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join("\n\n")
        )
    }
}

#[derive(Debug, Clone)]
pub struct Table {
    header: TableHeader,
    rows: Vec<TableRow>,
}

impl Table {
    pub fn new(header: TableHeader, rows: Vec<TableRow>) -> Self {
        assert!(!header.cells.is_empty());
        assert!(!rows.is_empty());
        Self { header, rows }
    }
}

impl Display for Table {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = self.header.to_string();
        let separators =
            vec![COLUMN_SEPARATOR; self.header.cells.len()].join(&format!(" {} ", TABLE_SEPARATOR));
        let rows = self
            .rows
            .iter()
            .map(|row| row.to_string())
            .collect::<Vec<String>>();
        write!(
            f,
            "{}\n{} {} {}\n{}",
            header,
            TABLE_SEPARATOR,
            separators,
            TABLE_SEPARATOR,
            rows.join("\n")
        )
    }
}

type Cell = String;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableHeader {
    cells: Vec<Cell>,
}

impl TableHeader {
    pub fn new(cells: Vec<Cell>) -> Self {
        Self { cells }
    }
}

impl Display for TableHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "| {} |", self.cells.join(" | "))
    }
}

/// Represents a row in the markdown table
#[derive(Debug, Clone)]
pub struct TableRow {
    cells: Vec<Cell>,
    values: Vec<Cell>,
}

impl TableRow {
    pub fn new(cells: Vec<Cell>, values: Vec<Cell>) -> Self {
        Self { cells, values }
    }
}

impl Display for TableRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ", TABLE_SEPARATOR)?;
        write!(f, "{}", &self.cells.join(&format!(" {} ", TABLE_SEPARATOR)))?;
        write!(f, " {} ", TABLE_SEPARATOR)?;
        write!(
            f,
            " {} ",
            &self.values.join(&format!(" {} ", TABLE_SEPARATOR))
        )?;
        write!(f, " {}", TABLE_SEPARATOR)
    }
}

/// Formats a number for display, showing integers with thousands separators and floats with 2 decimal places
pub fn format_metric_value(metric: &Metric) -> String {
    if metric.diff.is_some() {
        format!(
            "{value_str} <span style=\"color: {color}\">({diff} [{diff_percentage}%])</span> ",
            diff = format_number(metric.diff.unwrap()),
            diff_percentage = format_number(metric.diff_percentage.unwrap()),
            value_str = format_number(metric.value),
            color = {
                if metric.diff.unwrap() > 0.0 {
                    "red"
                } else {
                    "green"
                }
            },
        )
    } else {
        format_number(metric.value)
    }
}

fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        let int_value = value as i64;
        int_value.to_formatted_string(&Locale::en)
    } else if value.is_nan() {
        String::default()
    } else {
        format!("{:.2}", value)
    }
}
