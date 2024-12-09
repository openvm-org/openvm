use std::path::PathBuf;

use clap::Parser;
use eyre::Result;

use super::build::BuildArgs;
use crate::util::ProofMode;

#[derive(Parser)]
#[command(name = "keygen", about = "Generate a proving key")]
pub struct KeygenCmd {
    #[clap(long, action)]
    output: PathBuf,

    #[clap(value_enum)]
    mode: ProofMode,

    #[clap(flatten)]
    build_args: BuildArgs,
}

impl KeygenCmd {
    pub fn run(&self) -> Result<()> {
        todo!()
    }
}
