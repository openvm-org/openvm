use std::fs;

use afs_test_utils::page_config::PageConfig;
use clap::Parser;
use color_eyre::eyre::Result;

pub mod benchmark;
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
