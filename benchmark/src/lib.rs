use chrono::Local;
use lazy_static::lazy_static;

pub mod cli;
pub mod commands;
pub mod utils;

pub const TABLE_ID: &str = "0xfade";
pub const TMP_FOLDER: &str = "benchmark/tmp";
lazy_static! {
    pub static ref TMP_TRACING_LOG: String = TMP_FOLDER.to_string() + "/tracing.log";
    pub static ref DB_FILE_PATH: String = TMP_FOLDER.to_string() + "/db.mockdb";
    pub static ref AFI_FILE_PATH: String = TMP_FOLDER.to_string() + "/instructions.afi";
    pub static ref DEFAULT_OUTPUT_FILE: String = format!(
        "benchmark/output/{}.csv",
        Local::now().format("%Y%m%d-%H%M%S")
    );
}
