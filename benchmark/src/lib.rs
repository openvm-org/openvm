use lazy_static::lazy_static;

pub mod cli;
pub mod commands;
pub mod output_writer;
pub mod random_table;

pub const TABLE_ID: &str = "0xfade";
pub const TMP_FOLDER: &str = "bin/common/data/tmp";
lazy_static! {
    pub static ref DB_FILE_PATH: String = TMP_FOLDER.to_string() + "/db.mockdb";
}
