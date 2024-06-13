use alloy_primitives::U256;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input_instructions::AfsInputInstructions, afs_interface::AfsInterface, mock_db::MockDb,
    table::types::TableMetadata,
};

#[derive(Debug, Parser)]
pub struct WriteCommand {
    #[arg(
        long = "afi-file",
        short = 'f',
        help = "The .afi file input",
        required = true
    )]
    pub afi_file_path: String,

    #[arg(long = "table-id", short = 't', help = "The table ID", required = true)]
    pub table_id: String,

    #[arg(
        long = "db-file",
        short = 'd',
        help = "Mock DB file input (default: new empty DB)",
        required = false
    )]
    pub db_file_path: Option<String>,
}

/// `mock read` subcommand
impl WriteCommand {
    /// Execute the `mock read` command
    pub fn execute(self) -> Result<()> {}
}
