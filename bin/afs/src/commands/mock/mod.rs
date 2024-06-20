mod afi;
mod read;
mod write;

use afs_test_utils::page_config::PageConfig;
use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;

#[derive(Debug, Parser)]
pub struct MockCommand {
    #[command(subcommand)]
    pub command: MockSubcommands,
}

#[derive(Subcommand, Debug)]
pub enum MockSubcommands {
    /// `afi` subcommand
    Afi(afi::AfiCommand),

    /// `read` subcommand
    Read(read::ReadCommand),

    /// `write` subcommand
    Write(write::WriteCommand),
}

impl MockCommand {
    pub fn execute(&self, config: &PageConfig) -> Result<()> {
        match &self.command {
            MockSubcommands::Afi(afi) => {
                let cmd = afi::AfiCommand {
                    afi_file_path: afi.afi_file_path.clone(),
                    silent: afi.silent,
                };
                cmd.execute()
            }
            MockSubcommands::Read(read) => {
                let cmd = read::ReadCommand {
                    db_file_path: read.db_file_path.clone(),
                    table_id: read.table_id.clone(),
                    silent: read.silent,
                };
                cmd.execute(config)
            }
            MockSubcommands::Write(write) => {
                let cmd = write::WriteCommand {
                    afi_file_path: write.afi_file_path.clone(),
                    db_file_path: write.db_file_path.clone(),
                    output_db_file_path: write.output_db_file_path.clone(),
                    silent: write.silent,
                };
                cmd.execute(config)
            }
        }
    }
}
