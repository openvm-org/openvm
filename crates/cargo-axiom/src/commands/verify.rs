use std::path::PathBuf;

use clap::Parser;
use eyre::Result;

use super::build::BuildArgs;
use crate::util::ProofMode;

#[derive(Parser)]
#[command(name = "verify", about = "Verify a proof")]
pub struct VerifyCmd {
    #[clap(long, action)]
    proof: PathBuf,

    #[clap(value_enum)]
    mode: ProofMode,

    #[clap(flatten)]
    build_args: BuildArgs,
}

impl VerifyCmd {
    pub fn run(&self) -> Result<()> {
        todo!()
    }
}
