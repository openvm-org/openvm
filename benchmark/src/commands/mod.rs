use std::{collections::HashMap, fs, time::Instant};

use afs_test_utils::page_config::PageConfig;
use chrono::Local;
use clap::Parser;
use color_eyre::eyre::Result;

use crate::{
    utils::{
        config_gen::get_configs,
        output_writer::{
            default_output_filename, save_afi_to_new_db, write_csv_header, write_csv_line,
        },
        tracing::{clear_tracing_log, extract_event_data_from_log, extract_timing_data_from_log},
    },
    AFI_FILE_PATH, DB_FILE_PATH, TABLE_ID, TMP_FOLDER, TMP_TRACING_LOG,
};

pub mod predicate;
pub mod rw;

#[derive(Debug, Parser)]
pub struct CommonCommands {
    #[arg(
        long = "config-folder",
        short = 'c',
        help = "Runs the benchmark for all .toml PageConfig files in the folder",
        required = false
    )]
    pub config_folder: Option<String>,

    #[arg(
        long = "output-file",
        short = 'o',
        help = "Save output to this path (default: benchmark/output/<date>.csv)",
        required = false
    )]
    pub output_file: Option<String>,

    #[arg(
        long = "silent",
        short = 's',
        help = "Run the benchmark in silent mode",
        required = false
    )]
    pub silent: bool,
}

/// Function for setting up the benchmark
pub fn benchmark_setup(
    benchmark_name: String,
    config_folder: Option<String>,
    output_file: Option<String>,
) -> (Vec<PageConfig>, String) {
    // Generate/Parse config(s)
    let configs = get_configs(config_folder);

    // Create tmp folder
    let _ = fs::create_dir_all(TMP_FOLDER);

    // Write .csv file
    let output_file = output_file
        .clone()
        .unwrap_or(default_output_filename(benchmark_name.clone()));

    println!("Output file: {}", output_file.clone());
    write_csv_header(output_file.clone()).unwrap();

    (configs, output_file)
}

/// Function for setting up and running benchmarks. Takes in a `benchmark_fn` that runs the benchmarks themselves
/// as well as an `afi_gen_fn` that generates AFI files in a predetermined fashion.
pub fn benchmark_execute(
    benchmark_name: String,
    scenario: String,
    common: CommonCommands,
    extra_data: String,
    benchmark_fn: fn(&PageConfig, String) -> Result<()>,
    afi_gen_fn: fn(&PageConfig, String, String, usize, usize) -> Result<()>,
) -> Result<()> {
    println!("Executing [{}: {}] benchmark...", benchmark_name, scenario);

    let (configs, output_file) = benchmark_setup(
        benchmark_name.clone(),
        common.config_folder.clone(),
        common.output_file.clone(),
    );
    let configs_len = configs.len();
    println!("Output file: {}", output_file.clone());

    let (percent_reads, percent_writes) = {
        let parts: Vec<&str> = extra_data.split_whitespace().collect();
        let percent_reads = parts[0].parse::<usize>()?;
        let percent_writes = parts[1].parse::<usize>()?;
        (percent_reads, percent_writes)
    };

    // Run benchmark for each config
    for (idx, config) in configs.iter().rev().enumerate() {
        let timestamp = Local::now().format("%H:%M:%S");
        println!(
            "[{}] Running config {:?}: {} of {}",
            timestamp,
            config.generate_filename(),
            idx + 1,
            configs_len
        );

        clear_tracing_log(TMP_TRACING_LOG.as_str())?;

        // Generate AFI file
        let generate_afi_instant = Instant::now();
        afi_gen_fn(
            config,
            TABLE_ID.to_string(),
            AFI_FILE_PATH.to_string(),
            percent_reads,
            percent_writes,
        )?;
        let generate_afi_duration = generate_afi_instant.elapsed();
        println!("Setup: generate AFI duration: {:?}", generate_afi_duration);

        // Save AFI file data to database
        let save_afi_instant = Instant::now();
        save_afi_to_new_db(config, AFI_FILE_PATH.to_string(), DB_FILE_PATH.to_string())?;
        let save_afi_duration = save_afi_instant.elapsed();
        println!("Setup: save AFI to DB duration: {:?}", save_afi_duration);

        // Run the benchmark function
        benchmark_fn(config, extra_data.clone()).unwrap();

        let event_data = extract_event_data_from_log(
            TMP_TRACING_LOG.as_str(),
            &[
                "Total air width: preprocessed=",
                "Total air width: partitioned_main=",
                "Total air width: after_challenge=",
            ],
        )?;
        let timing_data = extract_timing_data_from_log(
            TMP_TRACING_LOG.as_str(),
            &[
                "Benchmark keygen: benchmark",
                "Benchmark cache: benchmark",
                "Benchmark prove: benchmark",
                "prove:Load page trace generation",
                "prove:Load page trace commitment",
                "Generate ops_sender trace",
                "prove:Prove trace commitment",
                "Benchmark verify: benchmark",
            ],
        )?;

        println!("Config: {:?}", config);
        println!("Event data: {:?}", event_data);
        println!("Timing data: {:?}", timing_data);
        println!("Output file: {}", output_file.clone());

        let mut log_data: HashMap<String, String> = event_data;
        log_data.extend(timing_data);

        write_csv_line(
            output_file.clone(),
            benchmark_name.clone(),
            scenario.clone(),
            config,
            &log_data,
        )?;
    }

    println!("Benchmark [{}: {}] completed.", benchmark_name, scenario);

    Ok(())
}

pub fn parse_config_folder(config_folder: String) -> Vec<PageConfig> {
    let mut configs = Vec::new();
    if let Ok(entries) = fs::read_dir(config_folder) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("toml") {
                let config = PageConfig::read_config_file(path.to_str().unwrap());
                configs.push(config);
            }
        }
    }
    configs
}
