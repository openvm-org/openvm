use clap::Parser;

use super::CommonCommands;

#[derive(Debug, Parser)]
pub struct OlapCommand {
    #[command(flatten)]
    pub common: CommonCommands,
}

impl OlapCommand {
    pub fn execute(&self) {
        println!("Executing OLAP benchmark...");
    }
}
