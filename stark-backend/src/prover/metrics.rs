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
            "Total F Cells: {} (excluding preprocessed)",
            self.total_cells
        )?;
        for trace_metrics in &self.per_air {
            writeln!(f, "{}", trace_metrics)?;
        }
        Ok(())
    }
}

impl Display for SingleTraceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Air: {}, Height: {}, Total F Cells: {}",
            self.air_name, self.height, self.total_cells
        )?;
        writeln!(f, "    {:?}", self.width)?;
        writeln!(f, "    {:?}", self.cells)?;
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
            let width = pk.vk.width().clone();
            let ext_degree = <SC::Challenge as AbstractExtensionField<Val<SC>>>::D;
            let cells = TraceCells {
                preprocessed: width.preprocessed.map(|w| w * height),
                partitioned_main: width.partitioned_main.iter().map(|w| w * height).collect(),
                after_challenge: width
                    .after_challenge
                    .iter()
                    .map(|w| w * height * ext_degree)
                    .collect(),
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
