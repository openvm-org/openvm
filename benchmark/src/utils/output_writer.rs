use std::{collections::HashMap, fs::OpenOptions};

use afs_test_utils::page_config::PageConfig;
use chrono::Local;
use color_eyre::eyre::Result;
use csv::{Writer, WriterBuilder};
use logical_interface::{afs_interface::AfsInterface, mock_db::MockDb};
use p3_util::ceil_div_usize;

use crate::config::benchmark_data::BenchmarkData;

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
