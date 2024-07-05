use afs_test_utils::page_config::PageConfig;
use clap::Parser;

pub mod olap;
pub mod rw;

#[derive(Debug, Parser)]
pub struct CommonCommands {
    #[arg(
        long = "config-files",
        short = 'c',
        help = "Comma-separated paths to config files",
        required = false,
        default_value = "config.toml"
    )]
    pub config_files: String,

    #[arg(
        long = "output-file",
        short = 'o',
        help = "Path to an output file",
        required = false,
        default_value = "output.csv"
    )]
    pub output_file: String,

    #[arg(
        long = "silent",
        short = 's',
        help = "Run the benchmark in silent mode",
        required = false
    )]
    pub silent: bool,
}

pub fn parse_configs(config_files: &String) -> Vec<PageConfig> {
    config_files
        .split(',')
        .map(PageConfig::read_config_file)
        .collect()
}
