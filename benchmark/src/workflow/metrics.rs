use std::{
    collections::BTreeMap,
    fmt::{self, Display},
};

use afs_stark_backend::prover::metrics::{format_number_with_underscores, TraceMetrics};
use serde::{Deserialize, Serialize};

/// Reusable struct for storing benchmark metrics
#[derive(Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics<CustomMetrics> {
    /// Benchmark name
    pub name: String,
    // Timings:
    pub total_prove_ms: f64,
    pub main_trace_gen_ms: f64,
    pub perm_trace_gen_ms: f64,
    pub calc_quotient_values_ms: f64,

    /// Trace metrics
    pub trace: TraceMetrics,

    /// Custom metrics
    pub custom: CustomMetrics,
}

// Implement the Display trait for BenchmarkMetrics to create a markdown table
impl<CustomMetrics: Display> Display for BenchmarkMetrics<CustomMetrics> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "## Benchmark for {}", self.name)?;
        // Write the markdown table header
        writeln!(
            f,
            "| Total Cells | Total Prove (ms) | Main Trace Gen (ms) | Perm Trace Gen (ms) | Calc Quotient Values (ms) | Rest of Prove (ms) |"
        )?;
        writeln!(
            f,
            "|-----------------------------|-----------------------|--------------------------|--------------------------|-----------------|----------------|"
        )?;

        // Write the metrics as a single row in the markdown table
        writeln!(
            f,
            "| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
            format_number_with_underscores(self.trace.total_cells),
            self.total_prove_ms,
            self.main_trace_gen_ms,
            self.perm_trace_gen_ms,
            self.calc_quotient_values_ms,
            self.total_prove_ms
                - self.main_trace_gen_ms
                - self.perm_trace_gen_ms
                - self.calc_quotient_values_ms
        )?;
        writeln!(f)?;

        writeln!(f, "### AIR metrics")?;
        writeln!(
            f,
            "| Name | Rows | Cells | Prep Cols | Main Cols | Perm Cols |"
        )?;
        writeln!(
            f,
            "|------|------|-------|-----------|-----------|-----------|"
        )?;
        for m in self.trace.per_air.iter() {
            writeln!(
                f,
                "| {:<20} | {:<10} | {:<11} | {:<5} | {:?} | {:?} |",
                m.air_name,
                format_number_with_underscores(m.height),
                format_number_with_underscores(m.total_cells),
                m.width.preprocessed.unwrap_or(0),
                m.width.partitioned_main,
                m.width.after_challenge,
            )?;
        }

        self.custom.fmt(f)?;

        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VmCustomMetrics {
    pub vm_metrics: BTreeMap<String, String>,
    pub opcode_counts: Vec<(String, usize)>,
    pub dsl_counts: Vec<(String, usize)>,
    pub opcode_trace_cells: Vec<(String, usize)>,
}

impl Display for VmCustomMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "<details>")?;
        writeln!(f, "<summary>")?;
        writeln!(f)?;
        writeln!(f, "### Custom VM metrics")?;
        writeln!(f)?;
        writeln!(f, "</summary>")?;
        writeln!(f)?;

        writeln!(f, "| Name | Value |")?;
        writeln!(f, "|------|-------|")?;
        for (name, value) in self.vm_metrics.iter() {
            writeln!(f, "| {:<20} | {:<10} |", name, value)?;
        }

        writeln!(f)?;
        writeln!(f, "#### Opcode metrics")?;
        writeln!(f, "| Name | Frequency | Trace Cells Contributed |")?;
        writeln!(f, "|------|-------|-----|")?;
        for (name, value) in self.opcode_counts.iter() {
            let cell_count = *self
                .opcode_trace_cells
                .iter()
                .find_map(|(k, v)| if k == name { Some(v) } else { None })
                .unwrap_or(&0);
            writeln!(f, "| {:<20} | {:<10} | {:<10} |", name, value, cell_count)?;
        }
        for (name, value) in self.opcode_trace_cells.iter() {
            if !self.opcode_counts.iter().any(|(k, _)| k == name) {
                // this should never happen
                writeln!(f, "| {:<20} | 0 | {:<10} |", name, value)?;
            }
        }

        writeln!(f)?;
        writeln!(f, "### DSL counts")?;
        writeln!(f, "How many opcodes each DSL instruction generates:")?;
        writeln!(f, "| Name | Count |")?;
        writeln!(f, "|------|-------|")?;
        for (name, value) in self.dsl_counts.iter() {
            writeln!(f, "| {:<20} | {:<10} |", name, value)?;
        }

        writeln!(f, "</details>")?;
        Ok(())
    }
}
