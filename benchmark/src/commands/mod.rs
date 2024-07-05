use clap::Parser;

pub mod olap;
pub mod rw;

#[derive(Debug, Parser)]
pub struct CommonCommands {
    #[arg(
        long = "config",
        short = 'c',
        help = "Path to a config file",
        required = false,
        default_value = "config.toml"
    )]
    pub config: String,

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
