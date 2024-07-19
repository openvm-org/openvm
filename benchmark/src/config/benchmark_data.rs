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

#[derive(Debug, Clone)]
pub struct BenchmarkData {
    pub sections: Vec<String>,
    pub headers: Vec<String>,
    pub event_tags: Vec<String>,
    pub timing_tags: Vec<String>,
}

#[derive(Debug, Clone)]
struct BenchmarkSetup {
    event_section: String,
    event_headers: Vec<String>,
    event_tags: Vec<String>,
    timing_section: String,
    timing_headers: Vec<String>,
    timing_tags: Vec<String>,
}

pub fn benchmark_data_predicate() -> BenchmarkData {
    let setup = BenchmarkSetup {
        event_section: "air width".to_string(),
        event_headers: ["preprocessed", "main", "challenge"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        event_tags: [
            "Total air width: preprocessed=",
            "Total air width: partitioned_main=",
            "Total air width: after_challenge=",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        timing_section: "timing (ms)".to_string(),
        timing_headers: [
            "keygen_time",
            "cache_time",
            "prove_load_trace_gen",
            "prove_load_trace_commit",
            "prove_commit",
            "prove_time_total",
            "verify_time",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        timing_tags: [
            "Benchmark keygen: benchmark",
            "Benchmark cache: benchmark",
            "prove:Load page trace generation",
            "prove:Load page trace commitment",
            "prove:Prove trace commitment",
            "Benchmark prove: benchmark",
            "Benchmark verify: benchmark",
        ]
        .iter()
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
        event_tags: [
            "Total air width: preprocessed=",
            "Total air width: partitioned_main=",
            "Total air width: after_challenge=",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        timing_section: "timing (ms)".to_string(),
        timing_headers: [
            "keygen_time",
            "cache_time",
            "prove_load_trace_gen",
            "prove_load_trace_commit",
            "prove_ops_sender_gen",
            "prove_commit",
            "prove_time_total",
            "verify_time",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        timing_tags: [
            "Benchmark keygen: benchmark",
            "Benchmark cache: benchmark",
            "prove:Load page trace generation",
            "prove:Load page trace commitment",
            "Generate ops_sender trace",
            "prove:Prove trace commitment",
            "Benchmark prove: benchmark",
            "Benchmark verify: benchmark",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
    };
    build_benchmark_data(setup)
}

fn build_benchmark_data(setup: BenchmarkSetup) -> BenchmarkData {
    assert!(
        setup.event_headers.len() == setup.event_tags.len(),
        "event_headers and event_tags must have the same length"
    );
    assert!(
        setup.timing_headers.len() == setup.timing_tags.len(),
        "timing_headers and timing_tags must have the same length"
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

    let event_tags = setup.event_tags;
    let timing_tags = setup.timing_tags;

    BenchmarkData {
        sections,
        headers,
        event_tags,
        timing_tags,
    }
}
