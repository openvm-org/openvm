mod afi;
mod read;
mod write;

use afs_test_utils::page_config::MultitierPageConfig;
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
    pub fn execute(&self, config: &MultitierPageConfig) -> Result<()> {
        match &self.command {
            MockSubcommands::Afi(afi) => {
                let cmd = afi::AfiCommand {
                    afi_file_path: afi.afi_file_path.clone(),
                    silent: afi.silent,
                };
                cmd.execute()
            }
            MockSubcommands::Read(read) => read.execute(config),
            MockSubcommands::Write(write) => write.execute(config),
        }
    }
}
