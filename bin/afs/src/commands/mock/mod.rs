mod afi;
mod interface;

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

    /// `interface` subcommand
    Interface(interface::InterfaceCommand),
}

impl MockCommand {
    pub fn execute(&self) -> Result<()> {
        match &self.command {
            MockSubcommands::Afi(afi) => {
                let cmd = afi::AfiCommand {
                    afi_file_path: afi.afi_file_path.clone(),
                    db_file_path: afi.db_file_path.clone(),
                };
                cmd.execute()
            }
            MockSubcommands::Interface(interface) => {
                let cmd = interface::InterfaceCommand {
                    afi_file_path: interface.afi_file_path.clone(),
                    db_file_path: interface.db_file_path.clone(),
                    table_id: interface.table_id.clone(),
                    print: interface.print,
                };
                cmd.execute()
            }
        }
    }
}
