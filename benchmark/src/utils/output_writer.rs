use std::{collections::HashMap, fs::OpenOptions};

use afs_test_utils::{
    config::EngineType,
    page_config::{PageConfig, PageMode},
};
use color_eyre::eyre::Result;
use csv::{Writer, WriterBuilder};
use logical_interface::{afs_interface::AfsInterface, mock_db::MockDb};
use p3_util::ceil_div_usize;
use serde::{Deserialize, Serialize};

/// Benchmark row for csv output
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BenchmarkRow {
    pub test_type: String,
    pub scenario: String,
    pub engine: EngineType,
    pub index_bytes: usize,
    pub data_bytes: usize,
    pub page_width: usize,
    pub height: usize,
    pub max_rw_ops: usize,
    pub bits_per_fe: usize,
    pub mode: PageMode,
    pub log_blowup: usize,
    pub num_queries: usize,
    pub pow_bits: usize,
    /// Total width of preprocessed AIR
    pub preprocessed: usize,
    /// Total width of partitioned main AIR
    pub main: usize,
    /// Total width of after challenge AIR
    pub challenge: usize,
    /// Keygen time: Time to generate keys
    pub keygen_time: String,
    /// Cache time: Time to generate cached trace
    pub cache_time: String,
    /// Prove: Time to generate load_page_and_ops trace
    pub prove_load_trace_gen: String,
    /// Prove: Time to commit load_page_and_ops trace
    pub prove_load_trace_commit: String,
    /// Prove: Time to generate trace
    pub prove_generate: String,
    /// Prove: Time to commit trace
    pub prove_commit: String,
    /// Prove time: Total time to generate the proof (inclusive of all prove timing items above)
    pub prove_time: String,
    /// Verify time: Time to verify the proof
    pub verify_time: String,
}

pub fn save_afi_to_new_db(
    config: &PageConfig,
    afi_path: String,
    db_file_path: String,
) -> Result<()> {
    let mut db = MockDb::new();
    let mut interface = AfsInterface::new(config.page.index_bytes, config.page.data_bytes, &mut db);
    interface.load_input_file(afi_path.as_str())?;
    db.save_to_file(db_file_path.as_str())?;
    Ok(())
}

pub fn write_csv_header(path: String) -> Result<()> {
    let mut writer = Writer::from_path(path)?;

    // sections
    writer.write_record(&vec![
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
        "air width",
        "",
        "",
        "timing",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ])?;

    // headers
    writer.write_record(&vec![
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
        "preprocessed",
        "main",
        "challenge",
        "keygen_time",
        "cache_time",
        "prove_load_trace_gen",
        "prove_load_trace_commit",
        "prove_generate",
        "prove_commit",
        "prove_time",
        "verify_time",
    ])?;

    writer.flush()?;
    Ok(())
}

pub fn write_csv_line(
    path: String,
    test_type: String,
    config: &PageConfig,
    log_data: &HashMap<String, String>,
    percent_reads: usize,
    percent_writes: usize,
) -> Result<()> {
    let file = OpenOptions::new().append(true).open(path).unwrap();
    let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);

    let scenario = format!("{}%r/{}%w", percent_reads, percent_writes);
    let bytes_divisor = ceil_div_usize(config.page.bits_per_fe, 8);
    let idx_len = ceil_div_usize(config.page.index_bytes, bytes_divisor);
    let data_len = ceil_div_usize(config.page.data_bytes, bytes_divisor);
    let page_width = 1 + idx_len + data_len;
    let row = BenchmarkRow {
        test_type,
        scenario,
        engine: config.stark_engine.engine,
        index_bytes: config.page.index_bytes,
        data_bytes: config.page.data_bytes,
        page_width,
        height: config.page.height,
        max_rw_ops: config.page.max_rw_ops,
        bits_per_fe: config.page.bits_per_fe,
        mode: config.page.mode.clone(),
        log_blowup: config.fri_params.log_blowup,
        num_queries: config.fri_params.num_queries,
        pow_bits: config.fri_params.proof_of_work_bits,
        preprocessed: log_data
            .get("Total air width: preprocessed=")
            .unwrap()
            .parse::<usize>()?,
        main: log_data
            .get("Total air width: partitioned_main=")
            .unwrap()
            .parse::<usize>()?,
        challenge: log_data
            .get("Total air width: after_challenge=")
            .unwrap()
            .parse::<usize>()?,
        keygen_time: log_data.get("ReadWrite keygen").unwrap().to_owned(),
        cache_time: log_data.get("ReadWrite cache").unwrap().to_owned(),
        prove_load_trace_gen: log_data
            .get("prove:Load page trace generation: afs_chips::page_rw_checker::page_controller")
            .unwrap()
            .to_owned(),
        prove_load_trace_commit: log_data
            .get("prove:Load page trace commitment: afs_chips::page_rw_checker::page_controller")
            .unwrap()
            .to_owned(),
        prove_generate: log_data.get("Prove.generate_trace").unwrap().to_owned(),
        prove_commit: log_data
            .get("prove:Prove trace commitment")
            .unwrap()
            .to_owned(),
        prove_time: log_data.get("ReadWrite prove").unwrap().to_owned(),
        verify_time: log_data.get("ReadWrite verify").unwrap().to_owned(),
    };

    writer.serialize(row)?;
    writer.flush()?;
    Ok(())
}

pub fn display_output(data: String) {
    println!("{}", data);
}
