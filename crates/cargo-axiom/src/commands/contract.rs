use std::path::PathBuf;

use clap::Parser;
use eyre::Result;

use super::build::BuildArgs;

#[derive(Parser)]
#[command(name = "contract", about = "Generate final SNARK verifier contract")]
pub struct ContractCmd {
    #[clap(long, action)]
    output: PathBuf,

    #[clap(flatten)]
    build_args: BuildArgs,
}

impl ContractCmd {
    pub fn run(&self) -> Result<()> {
        todo!()
    }
}
