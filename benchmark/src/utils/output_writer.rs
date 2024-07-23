use std::{collections::HashMap, fs::OpenOptions};

use afs_test_utils::{
    config::EngineType,
    page_config::{MultitierPageConfig, PageConfig, PageMode},
};
use chrono::Local;
use color_eyre::eyre::Result;
use csv::{Writer, WriterBuilder};
use logical_interface::{afs_interface::AfsInterface, mock_db::MockDb};
use p3_util::ceil_div_usize;

use crate::config::benchmark_data::BenchmarkData;

/// Benchmark row for csv output
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MultitierBenchmarkRow {
    pub test_type: String,
    pub scenario: String,
    pub engine: EngineType,
    pub index_bytes: usize,
    pub data_bytes: usize,
    pub page_width: usize,
    pub leaf_height: usize,
    pub internal_height: usize,
    pub init_leaf_cap: usize,
    pub init_internal_cap: usize,
    pub final_leaf_cap: usize,
    pub final_internal_cap: usize,
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
    /// Prove: Time to generate load_page_and_ops trace and to commit
    pub prove_load_trace_gen_and_commit: String,
    /// Prove: Time to generate trace
    pub prove_generate: String,
    /// Prove: Time to commit trace
    pub prove_commit: String,
    /// Prove time: Total time to generate the proof (inclusive of all prove timing items above)
    pub prove_time: String,
    /// Verify time: Time to verify the proof
    pub verify_time: String,
    /// Page BTree Update time: Time to update Page BTree
    pub page_btree_updates_time: String,
    /// Page BTree Commit to Disk Time: Time to commit data to disk
    pub page_btree_commit_to_disk_time: String,
    /// Page BTree Load time: Time to load traces and prover trace data
    pub page_btree_load_time: String,
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

pub fn default_output_filename(benchmark_name: String) -> String {
    format!(
        "benchmark/output/{}-{}.csv",
        benchmark_name,
        Local::now().format("%Y%m%d-%H%M%S")
    )
}

pub fn write_csv_header(path: String, sections: Vec<String>, headers: Vec<String>) -> Result<()> {
    let mut writer = Writer::from_path(path)?;
    writer.write_record(&sections)?;
    writer.write_record(&headers)?;
    writer.flush()?;
    Ok(())
}

pub fn write_csv_line(
    path: String,
    test_type: String,
    scenario: String,
    config: &PageConfig,
    benchmark_data: &BenchmarkData,
    log_data: &HashMap<String, String>,
) -> Result<()> {
    let file = OpenOptions::new().append(true).open(path).unwrap();
    let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);

    let bytes_divisor = ceil_div_usize(config.page.bits_per_fe, 8);
    let idx_len = ceil_div_usize(config.page.index_bytes, bytes_divisor);
    let data_len = ceil_div_usize(config.page.data_bytes, bytes_divisor);
    let page_width = 1 + idx_len + data_len;
    let mut row = vec![
        test_type,
        scenario,
        config.stark_engine.engine.to_string(),
        config.page.index_bytes.to_string(),
        config.page.data_bytes.to_string(),
        page_width.to_string(),
        config.page.height.to_string(),
        config.page.max_rw_ops.to_string(),
        config.page.bits_per_fe.to_string(),
        config.page.mode.clone().to_string(),
        config.fri_params.log_blowup.to_string(),
        config.fri_params.num_queries.to_string(),
        config.fri_params.proof_of_work_bits.to_string(),
    ];
    let mut event_data = benchmark_data
        .event_filters
        .clone()
        .iter()
        .map(|tag| log_data.get(tag).unwrap_or(&"-".to_string()).to_owned())
        .collect::<Vec<String>>();
    row.append(&mut event_data);
    let mut timing_data = benchmark_data
        .timing_filters
        .clone()
        .iter()
        .map(|tag| log_data.get(tag).unwrap_or(&"-".to_string()).to_owned())
        .collect::<Vec<String>>();
    row.append(&mut timing_data);

    writer.serialize(&row)?;
    writer.flush()?;
    Ok(())
}

pub fn write_multitier_csv_line(
    path: String,
    test_type: String,
    scenario: String,
    config: &MultitierPageConfig,
    log_data: &HashMap<String, String>,
) -> Result<()> {
    let file = OpenOptions::new().append(true).open(path).unwrap();
    let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);

    let bytes_divisor = ceil_div_usize(config.page.bits_per_fe, 8);
    let idx_len = ceil_div_usize(config.page.index_bytes, bytes_divisor);
    let data_len = ceil_div_usize(config.page.data_bytes, bytes_divisor);
    let page_width = 1 + idx_len + data_len;
    let row = MultitierBenchmarkRow {
        test_type,
        scenario,
        engine: config.stark_engine.engine,
        index_bytes: config.page.index_bytes,
        data_bytes: config.page.data_bytes,
        page_width,
        leaf_height: config.page.leaf_height,
        internal_height: config.page.internal_height,
        init_leaf_cap: config.tree.init_leaf_cap,
        init_internal_cap: config.tree.init_internal_cap,
        final_leaf_cap: config.tree.final_leaf_cap,
        final_internal_cap: config.tree.final_internal_cap,
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
        prove_load_trace_gen_and_commit: log_data
            .get("prove:Load page trace generation: afs_chips::multitier_page_rw_checker::page_controller")
            .unwrap()
            .to_owned(),
        prove_generate: log_data.get("Prove.generate_trace").unwrap().to_owned(),
        prove_commit: log_data
            .get("prove:Prove trace commitment")
            .unwrap()
            .to_owned(),
        prove_time: log_data.get("ReadWrite prove").unwrap().to_owned(),
        verify_time: log_data.get("ReadWrite verify").unwrap().to_owned(),
        page_btree_updates_time: log_data.get("Page BTree Updates").unwrap().to_owned(),
        page_btree_commit_to_disk_time: log_data.get("Page BTree Commit to Disk").unwrap().to_owned(),
        page_btree_load_time: log_data.get("Page BTree Load Traces and Prover Data").unwrap().to_owned(),
    };

    writer.serialize(row)?;
    writer.flush()?;
    Ok(())
}

pub fn display_output(data: String) {
    println!("{}", data);
}

pub fn write_multitier_csv_header(path: String) -> Result<()> {
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
        "preprocessed",
        "main",
        "challenge",
        "keygen_time",
        "prove_load_trace_gen_and_commit",
        "prove_generate",
        "prove_commit",
        "prove_time",
        "verify_time",
        "page_btree_updates_time",
        "page_btree_commit_to_disk_time",
        "page_btree_load_time",
    ])?;

    writer.flush()?;
    Ok(())
}
