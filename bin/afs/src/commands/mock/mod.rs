mod afi;
mod describe;
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

    /// describe all tables in the mock database
    Describe(describe::DescribeCommand),
}

impl MockCommand {
    pub fn execute(&self, config: &PageConfig) -> Result<()> {
        match &self.command {
            MockSubcommands::Afi(cmd) => cmd.execute(),
            MockSubcommands::Read(cmd) => cmd.execute(config),
            MockSubcommands::Write(cmd) => cmd.execute(config),
            MockSubcommands::Describe(cmd) => cmd.execute(),
        }
    }
}
