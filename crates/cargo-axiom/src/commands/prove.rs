use std::path::PathBuf;

use clap::Parser;
use eyre::Result;

use super::build::BuildArgs;
use crate::util::ProofMode;

#[derive(Parser)]
#[command(name = "prove", about = "Generate a program proof")]
pub struct ProveCmd {
    #[clap(long, action)]
    proof: PathBuf,

    #[clap(value_enum)]
    mode: ProofMode,

    #[clap(flatten)]
    build_args: BuildArgs,
}

impl ProveCmd {
    pub fn run(&self) -> Result<()> {
        todo!()
    }
}
