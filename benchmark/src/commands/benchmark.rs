use std::{collections::HashMap, fs, path::Path, time::Instant};

use afs_test_utils::page_config::{MultitierPageConfig, PageConfig};
use chrono::Local;
use color_eyre::eyre::Result;

use crate::{
    config::{
        benchmark_data::BenchmarkData,
        config_gen::{get_configs, get_multitier_configs},
    },
    utils::{
        output_writer::{
            default_output_filename, multitier_page_config_to_row, page_config_to_row,
            save_afi_to_new_db, write_csv_header, write_csv_line,
        },
        tracing::{clear_tracing_log, extract_event_data_from_log, extract_timing_data_from_log},
    },
    AFI_FILE_PATH, DB_FILE_PATH, MULTITIER_TABLE_ID, TABLE_ID, TMP_FOLDER, TMP_TRACING_LOG,
};

use super::CommonCommands;

/// Function for setting up the benchmark
pub fn benchmark_setup(
    benchmark_name: String,
    config_folder: Option<String>,
    output_file: Option<String>,
    benchmark_data: &BenchmarkData,
) -> (Vec<PageConfig>, String) {
    // Generate/Parse config(s)
    let configs = get_configs(config_folder);

    // Create tmp folder
    let _ = fs::create_dir_all(TMP_FOLDER);

    // Write .csv file
    let output_file = output_file
        .clone()
        .unwrap_or(default_output_filename(benchmark_name.clone()));
    // Extract the directory path from the output file path
    if let Some(output_directory) = Path::new(&output_file).parent() {
        // Create the directory and any necessary parent directories
        fs::create_dir_all(output_directory).unwrap();
    }

    println!("Output file: {}", output_file.clone());
    write_csv_header(
        output_file.clone(),
        benchmark_data.sections.clone(),
        benchmark_data.headers.clone(),
    )
    .unwrap();

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
    benchmark_data_fn: fn() -> BenchmarkData,
    afi_gen_fn: fn(&PageConfig, String, String, usize, usize) -> Result<()>,
) -> Result<()> {
    println!("Executing [{}: {}] benchmark...", benchmark_name, scenario);

    let benchmark_data = benchmark_data_fn();
    let (configs, output_file) = benchmark_setup(
        benchmark_name.clone(),
        common.config_folder.clone(),
        common.output_file.clone(),
        &benchmark_data,
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
            benchmark_data.event_filters.clone(),
        )?;
        let timing_data = extract_timing_data_from_log(
            TMP_TRACING_LOG.as_str(),
            benchmark_data.timing_filters.clone(),
        )?;

        println!("Config: {:?}", config);
        println!("Event data: {:?}", event_data);
        println!("Timing data: {:?}", timing_data);
        println!("Output file: {}", output_file.clone());

        let mut log_data: HashMap<String, String> = event_data;
        log_data.extend(timing_data);

        let init_row = page_config_to_row(benchmark_name.clone(), scenario.clone(), config);
        write_csv_line(output_file.clone(), init_row, &benchmark_data, &log_data)?;
    }

    println!("Benchmark [{}: {}] completed.", benchmark_name, scenario);

    Ok(())
}

/// Function for setting up the benchmark
pub fn benchmark_multitier_setup(
    benchmark_name: String,
    config_folder: Option<String>,
    output_file: Option<String>,
    benchmark_data: &BenchmarkData,
) -> (Vec<MultitierPageConfig>, String) {
    // Generate/Parse config(s)
    let configs = get_multitier_configs(config_folder);

    // Create tmp folder
    let _ = fs::create_dir_all(TMP_FOLDER);

    // Write .csv file
    let output_file = output_file
        .clone()
        .unwrap_or(default_output_filename(benchmark_name.clone()));

    println!("Output file: {}", output_file.clone());
    write_csv_header(
        output_file.clone(),
        benchmark_data.sections.clone(),
        benchmark_data.headers.clone(),
    )
    .unwrap();

    (configs, output_file)
}

#[allow(clippy::too_many_arguments)]
pub fn benchmark_multitier_execute(
    benchmark_name: String,
    scenario: String,
    common: CommonCommands,
    extra_data: String,
    start_idx: usize,
    benchmark_fn: fn(&MultitierPageConfig, String) -> Result<()>,
    benchmark_data_fn: fn() -> BenchmarkData,
    afi_gen_fn: fn(&MultitierPageConfig, String, String) -> Result<()>,
) -> Result<()> {
    println!("Executing [{}: {}] benchmark...", benchmark_name, scenario);
    let benchmark_data = benchmark_data_fn();
    let (configs, output_file) = benchmark_multitier_setup(
        benchmark_name.clone(),
        common.config_folder.clone(),
        common.output_file.clone(),
        &benchmark_data,
    );
    let configs_len = configs.len();
    println!("Output file: {}", output_file.clone());

    // Run benchmark for each config
    for (idx, config) in configs.iter().rev().enumerate() {
        if idx < start_idx {
            continue;
        }
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
            MULTITIER_TABLE_ID.to_string(),
            AFI_FILE_PATH.to_string(),
        )?;
        let generate_afi_duration = generate_afi_instant.elapsed();
        println!("Setup: generate AFI duration: {:?}", generate_afi_duration);

        // Run the benchmark function
        benchmark_fn(config, extra_data.clone()).unwrap();

        let event_data = extract_event_data_from_log(
            TMP_TRACING_LOG.as_str(),
            benchmark_data.event_filters.clone(),
        )?;
        let timing_data = extract_timing_data_from_log(
            TMP_TRACING_LOG.as_str(),
            benchmark_data.timing_filters.clone(),
        )?;

        println!("Config: {:?}", config);
        println!("Event data: {:?}", event_data);
        println!("Timing data: {:?}", timing_data);
        println!("Output file: {}", output_file.clone());

        let mut log_data: HashMap<String, String> = event_data;
        log_data.extend(timing_data);
        let init_row =
            multitier_page_config_to_row(benchmark_name.clone(), scenario.clone(), config);
        write_csv_line(output_file.clone(), init_row, &benchmark_data, &log_data)?;
    }

    println!("Benchmark [{}: {}] completed.", benchmark_name, scenario);

    Ok(())
}
