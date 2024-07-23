use lazy_static::lazy_static;

pub mod cli;
pub mod commands;
pub mod config;
pub mod utils;

pub const TABLE_ID: &str = "0xfade";
pub const MULTITIER_TABLE_ID: &str = "BIG_TREE";
pub const TMP_FOLDER: &str = "benchmark/tmp";
lazy_static! {
    pub static ref TMP_TRACING_LOG: String = TMP_FOLDER.to_string() + "/_tracing.log";
    pub static ref DB_FILE_PATH: String = TMP_FOLDER.to_string() + "/db.mockdb";
    pub static ref AFI_FILE_PATH: String = TMP_FOLDER.to_string() + "/instructions.afi";
    pub static ref FILTER_FILE_PATH: String =
        String::from("benchmark/config/olap/filter_") + TABLE_ID + ".afo";
}
lazy_static! {
    pub static ref KEY_FOLDER: String = TMP_FOLDER.to_string() + "/keys";
    pub static ref DB_FOLDER: String = TMP_FOLDER.to_string() + "/db";
}
