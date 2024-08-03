use std::fmt::Display;

use itertools::Itertools;
use p3_field::AbstractExtensionField;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

use crate::keygen::types::{StarkProvingKey, TraceWidth};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceMetrics {
    pub per_air: Vec<SingleTraceMetrics>,
    /// Total base field cells from all traces, excludes preprocessed.
    pub total_cells: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SingleTraceMetrics {
    pub air_name: String,
    pub height: usize,
    /// The after challenge width is adjusted to be in terms of **base field** elements.
    pub width: TraceWidth,
    pub cells: TraceCells,
    /// Omitting preprocessed trace, the total base field cells from main and after challenge
    /// traces.
    pub total_cells: usize,
}

/// Trace cells, counted in terms of number of **base field** elements.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceCells {
    pub preprocessed: Option<usize>,
    pub partitioned_main: Vec<usize>,
    pub after_challenge: Vec<usize>,
}

impl Display for TraceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Total Cells: {} (excluding preprocessed)",
            format_number_with_underscores(self.total_cells)
        )?;
        for trace_metrics in &self.per_air {
            writeln!(f, "{}", trace_metrics)?;
        }
        Ok(())
    }
}

impl Display for SingleTraceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
           f,
            "{:<20} | Rows = {:<10} | Cells = {:<11} | Prep Cols = {:<5} | Main Cols = {:<5} | Perm Cols = {:<5}",
            self.air_name, format_number_with_underscores(self.height), format_number_with_underscores(self.total_cells), self.width.preprocessed.unwrap_or(0),
            format!("{:?}", self.width.partitioned_main),
            format!("{:?}",self.width.after_challenge),
        )?;
        Ok(())
    }
}

/// heights are the trace heights for each air
pub fn trace_metrics<SC: StarkGenericConfig>(
    pk: &[StarkProvingKey<SC>],
    heights: &[usize],
) -> TraceMetrics {
    let per_air: Vec<_> = pk
        .iter()
        .zip_eq(heights)
        .map(|(pk, &height)| {
            let air_name = pk.air_name.clone();
            let mut width = pk.vk.width().clone();
            let ext_degree = <SC::Challenge as AbstractExtensionField<Val<SC>>>::D;
            for w in &mut width.after_challenge {
                *w *= ext_degree;
            }
            let cells = TraceCells {
                preprocessed: width.preprocessed.map(|w| w * height),
                partitioned_main: width.partitioned_main.iter().map(|w| w * height).collect(),
                after_challenge: width.after_challenge.iter().map(|w| w * height).collect(),
            };
            let total_cells = cells
                .partitioned_main
                .iter()
                .chain(cells.after_challenge.iter())
                .sum::<usize>();
            SingleTraceMetrics {
                air_name,
                height,
                width,
                cells,
                total_cells,
            }
        })
        .collect();
    let total_cells = per_air.iter().map(|m| m.total_cells).sum();
    TraceMetrics {
        per_air,
        total_cells,
    }
}

pub fn format_number_with_underscores(n: usize) -> String {
    let num_str = n.to_string();
    let mut result = String::new();

    // Start adding characters from the end of num_str
    for (i, c) in num_str.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push('_');
        }
        result.push(c);
    }

    // Reverse the result to get the correct order
    result.chars().rev().collect()
}
