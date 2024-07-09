use std::{collections::HashMap, fs::OpenOptions};

use afs_test_utils::{
    config::EngineType,
    page_config::{PageConfig, PageMode},
};
use color_eyre::eyre::Result;
use csv::{Writer, WriterBuilder};
use logical_interface::{
    afs_interface::AfsInterface, mock_db::MockDb, table::types::TableMetadata,
};
use p3_util::ceil_div_usize;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BenchmarkRow {
    pub test_type: String,
    pub scenario: String,
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
    pub engine: EngineType,
    pub preprocessed: usize,
    pub main: usize,
    pub challenge: usize,
    pub keygen_time: String,
    pub cache_time: String,
    pub prove_generate: String,
    pub prove_commit: String,
    pub prove_time: String,
    pub verify_time: String,
}

pub fn save_afi_to_new_db(
    config: &PageConfig,
    afi_path: String,
    db_file_path: String,
) -> Result<()> {
    let table_metadata = TableMetadata::new(config.page.index_bytes, config.page.data_bytes);
    let mut db = MockDb::new(table_metadata);
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
        "stark engine",
        "air width",
        "",
        "",
        "timing",
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
        "engine",
        "preprocessed",
        "main",
        "challenge",
        "keygen_time",
        "cache_time",
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

    let max_writes = config.page.max_rw_ops * percent_writes / 100;
    let max_reads = config.page.max_rw_ops * percent_reads / 100;
    let scenario = format!("{}r/{}w", max_reads, max_writes);
    let bytes_divisor = ceil_div_usize(config.page.bits_per_fe, 8);
    let idx_len = ceil_div_usize(config.page.index_bytes, bytes_divisor);
    let data_len = ceil_div_usize(config.page.data_bytes, bytes_divisor);
    let page_width = 1 + idx_len + data_len;
    let row = BenchmarkRow {
        test_type,
        scenario,
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
        engine: config.stark_engine.engine,
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
