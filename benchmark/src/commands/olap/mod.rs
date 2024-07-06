use clap::Parser;
use color_eyre::eyre::Result;

use super::CommonCommands;

#[derive(Debug, Parser)]
pub struct OlapCommand {
    #[command(flatten)]
    pub common: CommonCommands,
}

impl OlapCommand {
    pub fn execute(&self) -> Result<()> {
        println!("Executing OLAP benchmark...");
        Ok(())
    }
}
