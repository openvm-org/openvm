pub mod cli;
pub mod commands;
pub mod operations;

pub const RANGE_CHECK_BITS: usize = 16;

pub const KEYS_FOLDER: &str = "bin/olap/tmp/keys";
pub const CACHE_FOLDER: &str = "bin/olap/tmp/cache";
