use clap::Parser;
use logical_interface::afs_input::operations_file::AfsOperationsFile;

pub mod cache;
pub mod keygen;
pub mod prove;
pub mod verify;

#[derive(Debug, Parser)]
pub struct CommonCommands {
    #[arg(
        long = "db-path",
        short = 'd',
        help = "The path to the database",
        required = true
    )]
    pub db_path: String,

    #[arg(
        long = "afo-path",
        short = 'f',
        help = "The path to the .afo file containing the OLAP commands",
        required = true
    )]
    pub afo_path: String,

    #[arg(
        long = "output-path",
        short = 'o',
        help = "The path to the output file",
        required = false
    )]
    pub output_path: Option<String>,

    #[arg(
        long = "silent",
        short = 's',
        help = "Don't print the output to stdout",
        required = false
    )]
    pub silent: bool,
}

pub fn parse_afo_file(afo_path: String) -> AfsOperationsFile {
    AfsOperationsFile::open(afo_path.clone())
}
