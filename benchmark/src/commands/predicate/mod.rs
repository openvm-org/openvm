use clap::Parser;
use color_eyre::eyre::Result;

use super::CommonCommands;

#[derive(Debug, Parser)]
pub struct PredicateCommand {
    #[command(flatten)]
    pub common: CommonCommands,
}

impl PredicateCommand {
    pub fn execute(&self) -> Result<()> {
        println!("Executing Predicate benchmark...");
        unimplemented!("WIP")
    }
}
