use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, Write},
    sync::Mutex,
};

use regex::Regex;
use tracing::Level;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{fmt::format::FmtSpan, layer::SubscriberExt};
use tracing_subscriber::{EnvFilter, Layer};

use color_eyre::eyre::{eyre, Result};

use crate::TMP_TRACING_LOG;

const TIME_PREFIX: &str = "time.busy=";

/// Sets up tracing to print to terminal and write to log file in parallel
pub fn setup_benchmark_tracing() -> WorkerGuard {
    let env_filter = EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .parse_lossy("benchmark=info,afs=info");

    let tmp_log = File::create(TMP_TRACING_LOG.as_str()).unwrap();
    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(Mutex::new(tmp_log))
        .with_ansi(false)
        .with_span_events(FmtSpan::CLOSE)
        .with_filter(env_filter);

    let env_filter2 = EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .from_env_lossy();
    let (non_blocking_writer, guard) = tracing_appender::non_blocking(std::io::stderr());
    let stderr_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_writer(non_blocking_writer)
        .with_filter(env_filter2);

    tracing_subscriber::registry()
        .with(file_layer)
        .with(stderr_layer)
        .init();

    guard
}

pub fn clear_tracing_log(file_path: &str) -> Result<()> {
    let mut file = File::create(file_path)?;
    file.write_all(b"")?;
    Ok(())
}

pub fn extract_timing_data_from_log(
    file_path: &str,
    filter_values: Vec<String>,
) -> Result<HashMap<String, String>> {
    let mut results: HashMap<String, String> = HashMap::new();
    if let Ok(file) = File::open(file_path) {
        for line in io::BufReader::new(file).lines() {
            let line = line.unwrap();
            for val in filter_values.iter() {
                if line.contains(val) {
                    if let Some(start) = line.find(TIME_PREFIX) {
                        let time_busy_start = start + TIME_PREFIX.len();
                        if let Some(end) = line[time_busy_start..].find(' ') {
                            let time_busy =
                                line[time_busy_start..time_busy_start + end].to_string();
                            let time_busy_string = convert_to_ms_string(&time_busy).unwrap();
                            results.insert(val.to_string(), time_busy_string);
                        }
                    }
                }
            }
        }
    }
    Ok(results)
}

pub fn extract_event_data_from_log(
    file_path: &str,
    filter_values: Vec<String>,
) -> Result<HashMap<String, String>> {
    let mut results: HashMap<String, String> = HashMap::new();
    if let Ok(file) = File::open(file_path) {
        for line in io::BufReader::new(file).lines() {
            let line = line.unwrap();
            for val in filter_values.iter() {
                if line.contains(val) {
                    if let Some(start) = line.find(val) {
                        let event_data_start = start + val.len();
                        if let Some(end) = line[event_data_start..].find(' ') {
                            let event_data =
                                line[event_data_start..event_data_start + end].to_string();
                            results.insert(val.to_string(), event_data);
                        }
                    }
                }
            }
        }
    }
    Ok(results)
}

fn convert_to_ms_string(time_string: &str) -> Result<String> {
    let time_unit_regex = Regex::new(r"(\d+\.?\d*)(ms|s|µs|ns)").unwrap();
    let captures = time_unit_regex.captures(time_string).unwrap();
    let time_value = captures.get(1).unwrap().as_str();
    let time_unit = captures.get(2).unwrap().as_str();
    let time_unit_float = match time_unit {
        "ms" => 1.0,
        "s" => 1000.0,
        "µs" => 0.001,
        "ns" => 0.000001,
        _ => return Err(eyre!("Invalid time unit: {}", time_unit)),
    };
    let time_in_ms = time_value.parse::<f64>()? * time_unit_float;
    Ok(format!("{:.2}", time_in_ms))
}
