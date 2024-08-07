use std::{collections::BTreeMap, fmt};

use afs_stark_backend::prover::metrics::{format_number_with_underscores, TraceMetrics};
use serde::{Deserialize, Serialize};

/// Reusable struct for storing benchmark metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CustomMetrics {
    pub vm_metrics: BTreeMap<String, String>,
    pub opcode_counts: Vec<(String, String)>,
    pub dsl_counts: Vec<(String, String)>,
    pub opcode_trace_cells: Vec<(String, String)>,
}

// Implement the Display trait for BenchmarkMetrics to create a markdown table
impl fmt::Display for BenchmarkMetrics {
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

        if !self.custom.vm_metrics.is_empty() {
            writeln!(f)?;
            writeln!(f, "### Custom metrics")?;
            writeln!(f, "| Name | Value |")?;
            writeln!(f, "|------|-------|")?;
            for (name, value) in self.custom.vm_metrics.iter() {
                writeln!(f, "| {:<20} | {:<10} |", name, value)?;
            }
        }

        if !self.custom.opcode_counts.is_empty() {
            writeln!(f)?;
            writeln!(f, "### Opcode counts")?;
            writeln!(f, "| Name | Count |")?;
            writeln!(f, "|------|-------|")?;
            for (name, value) in self.custom.opcode_counts.iter() {
                writeln!(f, "| {:<20} | {:<10} |", name, value)?;
            }
        }

        if !self.custom.dsl_counts.is_empty() {
            writeln!(f)?;
            writeln!(
                f,
                "### DSL counts - how many isa instructions each DSL instruction generates"
            )?;
            writeln!(f, "| Name | Count |")?;
            writeln!(f, "|------|-------|")?;
            for (name, value) in self.custom.dsl_counts.iter() {
                writeln!(f, "| {:<20} | {:<10} |", name, value)?;
            }
        }

        if !self.custom.opcode_trace_cells.is_empty() {
            writeln!(f)?;
            writeln!(f, "### Opcode trace cells")?;
            writeln!(f, "| Name | Count |")?;
            writeln!(f, "|------|-------|")?;
            for (name, value) in self.custom.opcode_trace_cells.iter() {
                writeln!(f, "| {:<20} | {:<10} |", name, value)?;
            }
        }
        Ok(())
    }
}
