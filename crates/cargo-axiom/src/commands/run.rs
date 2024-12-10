use std::path::PathBuf;

use clap::Parser;
use eyre::Result;

use super::build::BuildArgs;
use crate::util::Input;

#[derive(Parser)]
#[command(name = "run", about = "Run an axVM program")]
pub struct RunCmd {
    #[clap(long, value_parser)]
    input: Option<Input>,

    #[clap(long, action)]
    exe: PathBuf,

    #[clap(flatten)]
    build_args: BuildArgs,
}

impl RunCmd {
    pub fn run(&self) -> Result<()> {
        todo!()
    }
}
