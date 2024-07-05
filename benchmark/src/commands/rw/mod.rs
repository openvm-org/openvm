use clap::Parser;

use super::CommonCommands;

#[derive(Debug, Parser)]
pub struct RwCommand {
    #[command(flatten)]
    pub common: CommonCommands,
}

impl RwCommand {
    pub fn execute(&self) {
        println!("Executing Read/Write benchmark...");
    }
}
