use lazy_static::lazy_static;

lazy_static! {
    pub static ref CONFIG_SECTIONS: Vec<String> = [
        "benchmark",
        "",
        "stark engine",
        "page config",
        "",
        "",
        "",
        "",
        "",
        "",
        "fri params",
        "",
        "",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    pub static ref CONFIG_HEADERS: Vec<String> = [
        "test_type",
        "scenario",
        "engine",
        "index_bytes",
        "data_bytes",
        "page_width",
        "height",
        "max_rw_ops",
        "bits_per_fe",
        "mode",
        "log_blowup",
        "num_queries",
        "pow_bits",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
}

lazy_static! {
    pub static ref MULTITIER_CONFIG_SECTIONS: Vec<String> = [
        "benchmark",
        "",
        "stark engine",
        "page config",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "fri params",
        "",
        "",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    pub static ref MULTITIER_CONFIG_HEADERS: Vec<String> = [
        "test_type",
        "scenario",
        "engine",
        "index_bytes",
        "data_bytes",
        "page_width",
        "leaf_height",
        "internal_height",
        "init_leaf_cap",
        "init_internal_cap",
        "final_leaf_cap",
        "final_internal_cap",
        "max_rw_ops",
        "bits_per_fe",
        "mode",
        "log_blowup",
        "num_queries",
        "pow_bits",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
}

#[derive(Debug, Clone)]
pub struct BenchmarkData {
    pub sections: Vec<String>,
    pub headers: Vec<String>,
    pub event_filters: Vec<String>,
    pub timing_filters: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkSetup {
    /// Section name for events
    pub event_section: String,
    /// Headers for each event column
    pub event_headers: Vec<String>,
    /// Filter queries for the tracing logs for events
    pub event_filters: Vec<String>,
    /// Section name for timing
    pub timing_section: String,
    /// Headers for each timing column
    pub timing_headers: Vec<String>,
    /// Filter queries for the tracing logs for timing
    pub timing_filters: Vec<String>,
}

/// Descriptors for common timings in the stark-backend
pub const BACKEND_TIMING_HEADERS: &[&str] = &[
    "Prove: Gen permutation traces",
    "Prove: Commit permutation traces",
    "Prove: Compute quotient values",
    "Prove: Commit to quotient poly",
    "Prove: FRI opening proofs",
];
/// Timers from stark-backend
pub const BACKEND_TIMING_FILTERS: &[&str] = &[
    "prove:MultiTraceStarkProver::prove:generate permutation traces",
    "prove:MultiTraceStarkProver::prove:commit to permutation traces",
    "prove:prove_raps_with_committed_traces:compute quotient values",
    "prove:prove_raps_with_committed_traces:commit to quotient poly",
    "prove:prove_raps_with_committed_traces:FRI opening proofs",
];

/// Format for Predicate benchmark
pub fn benchmark_data_predicate() -> BenchmarkData {
    let setup = BenchmarkSetup {
        event_section: "air width".to_string(),
        event_headers: ["preprocessed", "main", "challenge"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        event_filters: [
            "Total air width: preprocessed=",
            "Total air width: partitioned_main=",
            "Total air width: after_challenge=",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        timing_section: "timing (ms)".to_string(),
        timing_headers: [
            "Keygen time",
            "Cache time",
            "Prove: Load trace gen",
            "Prove: Load trace commit",
            "Prove: Main commit",
        ]
        .iter()
        .chain(BACKEND_TIMING_HEADERS)
        .chain(&["Prove time (total)", "Verify time"])
        .map(|s| s.to_string())
        .collect(),
        timing_filters: [
            "Benchmark keygen: benchmark",
            "Benchmark cache: benchmark",
            "prove:Load page trace generation",
            "prove:Load page trace commitment",
            "prove:Prove trace commitment",
        ]
        .iter()
        .chain(BACKEND_TIMING_FILTERS)
        .chain(&["Benchmark prove: benchmark", "Benchmark verify: benchmark"])
        .map(|s| s.to_string())
        .collect(),
    };
    build_benchmark_data(setup)
}

pub fn benchmark_data_rw() -> BenchmarkData {
    let setup = BenchmarkSetup {
        event_section: "air width".to_string(),
        event_headers: ["preprocessed", "main", "challenge"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        event_filters: [
            "Total air width: preprocessed=",
            "Total air width: partitioned_main=",
            "Total air width: after_challenge=",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        timing_section: "timing (ms)".to_string(),
        timing_headers: [
            "Keygen time",
            "Cache time",
            "Prove: Load trace gen",
            "Prove: Load trace commit",
            "Prove: Main commit",
        ]
        .iter()
        .chain(BACKEND_TIMING_HEADERS)
        .chain(&["Prove time (total)", "Verify time"])
        .map(|s| s.to_string())
        .collect(),
        timing_filters: [
            "Benchmark keygen: benchmark",
            "Benchmark cache: benchmark",
            "prove:Load page trace generation",
            "prove:Load page trace commitment",
            "prove:Prove trace commitment",
        ]
        .iter()
        .chain(BACKEND_TIMING_FILTERS)
        .chain(&["Benchmark prove: benchmark", "Benchmark verify: benchmark"])
        .map(|s| s.to_string())
        .collect(),
    };
    build_benchmark_data(setup)
}

/// Format for ReadWrite benchmark
pub fn benchmark_data_multitier_rw() -> BenchmarkData {
    let setup = BenchmarkSetup {
        event_section: "air width".to_string(),
        event_headers: ["preprocessed", "main", "challenge"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        event_filters: [
            "Total air width: preprocessed=",
            "Total air width: partitioned_main=",
            "Total air width: after_challenge=",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        timing_section: "timing (ms)".to_string(),
        timing_headers: [
            "Keygen time",
            "Prove: Load trace gen and commit",
            "Prove: Tree DFS time",
            "Prove: Main Trace Generation",
            "Prove: Final Page Trace Generation",
            "Prove: Less Than Trace Generation",
            "Prove: Offline Checker Trace Generation",
            "Prove: Main commit",
        ]
        .iter()
        .chain(BACKEND_TIMING_HEADERS)
        .chain(&[
            "Prove time (total)",
            "Verify time",
            "Page BTree: Update Time",
            "Page BTree: Commit To Disk Time",
            "Page BTree: Load and Generate Output Time",
        ])
        .map(|s| s.to_string())
        .collect(),
        timing_filters: [
            "Benchmark keygen: benchmark",
            "prove:Load page trace generation",
            "prove:Load page trace generation:Tree DFS",
            "prove:Load page trace generation:Main Trace Generation",
            "prove:Load page trace generation:Main Trace Generation:Final Page Trace Generation",
            "prove:Load page trace generation:Main Trace Generation:Less Than Trace Generation",
            "prove:Load page trace generation:Ops Trace Generation",
            "prove:Prove trace commitment",
        ]
        .iter()
        .chain(BACKEND_TIMING_FILTERS)
        .chain(&[
            "Benchmark prove: benchmark",
            "Benchmark verify: benchmark",
            "Page BTree Updates",
            "Page BTree Commit to Disk",
            "Page BTree Load Traces and Prover Data, Generate Output Traces",
        ])
        .map(|s| s.to_string())
        .collect(),
    };
    build_multitier_benchmark_data(setup)
}

fn build_benchmark_data(setup: BenchmarkSetup) -> BenchmarkData {
    assert!(
        setup.event_headers.len() == setup.event_filters.len(),
        "event_headers and event_filters must have the same length"
    );
    assert!(
        setup.timing_headers.len() == setup.timing_filters.len(),
        "timing_headers and timing_filters must have the same length"
    );

    // Extend `section_events` and `section_timings` to the same length as `headers_events` and `headers_timings`, respectively
    let mut event_sections = vec![setup.event_section];
    event_sections.resize_with(setup.event_headers.len(), String::new);
    let mut timing_sections = vec![setup.timing_section];
    timing_sections.resize_with(setup.timing_headers.len(), String::new);

    // Build the sections vec
    let sections = [
        CONFIG_SECTIONS.as_slice(),
        &event_sections,
        &timing_sections,
    ]
    .iter()
    .flat_map(|s| s.iter())
    .cloned()
    .collect();

    // Build the headers vec
    let headers = CONFIG_HEADERS
        .as_slice()
        .iter()
        .chain(setup.event_headers.iter())
        .chain(setup.timing_headers.iter())
        .cloned()
        .collect();

    let event_filters = setup.event_filters;
    let timing_filters = setup.timing_filters;

    BenchmarkData {
        sections,
        headers,
        event_filters,
        timing_filters,
    }
}

fn build_multitier_benchmark_data(setup: BenchmarkSetup) -> BenchmarkData {
    assert!(
        setup.event_headers.len() == setup.event_filters.len(),
        "event_headers and event_filters must have the same length"
    );
    assert!(
        setup.timing_headers.len() == setup.timing_filters.len(),
        "timing_headers and timing_filters must have the same length"
    );

    // Extend `section_events` and `section_timings` to the same length as `headers_events` and `headers_timings`, respectively
    let mut event_sections = vec![setup.event_section];
    event_sections.resize_with(setup.event_headers.len(), String::new);
    let mut timing_sections = vec![setup.timing_section];
    timing_sections.resize_with(setup.timing_headers.len(), String::new);

    // Build the sections vec
    let sections = [
        MULTITIER_CONFIG_SECTIONS.as_slice(),
        &event_sections,
        &timing_sections,
    ]
    .iter()
    .flat_map(|s| s.iter())
    .cloned()
    .collect();

    // Build the headers vec
    let headers = MULTITIER_CONFIG_HEADERS
        .as_slice()
        .iter()
        .chain(setup.event_headers.iter())
        .chain(setup.timing_headers.iter())
        .cloned()
        .collect();

    let event_filters = setup.event_filters;
    let timing_filters = setup.timing_filters;

    BenchmarkData {
        sections,
        headers,
        event_filters,
        timing_filters,
    }
}
