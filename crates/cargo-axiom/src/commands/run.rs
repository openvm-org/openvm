use std::path::PathBuf;

use axvm_sdk::{
    config::{AppConfig, SdkVmConfig},
    fs::read_exe_from_file,
    Sdk,
};
use clap::Parser;
use eyre::Result;

use crate::util::{read_to_stdin, read_to_struct_toml, Input};

#[derive(Parser)]
#[command(name = "run", about = "Run an axVM program")]
pub struct RunCmd {
    #[clap(long, action, help = "Path to axVM executable")]
    exe: PathBuf,

    #[clap(long, action, help = "Path to app config TOML file")]
    config: PathBuf,

    #[clap(long, value_parser, help = "Input to axVM program")]
    input: Option<Input>,
}

impl RunCmd {
    pub fn run(&self) -> Result<()> {
        let exe = read_exe_from_file(&self.exe)?;
        let app_config: AppConfig<SdkVmConfig> = read_to_struct_toml(&self.config)?;
        let output = Sdk.execute(exe, app_config.app_vm_config, read_to_stdin(&self.input)?)?;
        println!("Execution output: {:?}", output);
        Ok(())
    }
}
