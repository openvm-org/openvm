use std::path::PathBuf;

use clap::Parser;
use eyre::Result;

use super::build::BuildArgs;

#[derive(Parser)]
#[command(name = "transpile", about = "Transpile an ELF into an axVM program")]
pub struct TranspileCmd {
    #[clap(long, action)]
    elf: PathBuf,

    #[clap(flatten)]
    build_args: BuildArgs,
}

impl TranspileCmd {
    pub fn run(&self) -> Result<()> {
        todo!()
    }
}
