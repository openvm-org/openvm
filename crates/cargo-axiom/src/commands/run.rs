use std::path::PathBuf;

use axvm_sdk::{config::SdkVmConfig, fs::read_exe_from_file, Sdk};
use clap::Parser;
use eyre::Result;

use crate::util::{read_to_stdin, read_to_struct_toml, Input};

#[derive(Parser)]
#[command(name = "run", about = "Run an axVM program")]
pub struct RunCmd {
    #[clap(long, action, help = "Path to axVM executable")]
    exe: PathBuf,

    #[clap(long, action, help = "Path to VM configuration TOML file")]
    vm_config: PathBuf,

    #[clap(long, value_parser, help = "Input to axVM program")]
    input: Option<Input>,
}

impl RunCmd {
    pub fn run(&self) -> Result<()> {
        let exe = read_exe_from_file(&self.exe)?;
        let vm_config: SdkVmConfig = read_to_struct_toml(&self.vm_config)?;
        Sdk.execute(exe, vm_config, read_to_stdin(&self.input)?)?;
        Ok(())
    }
}
