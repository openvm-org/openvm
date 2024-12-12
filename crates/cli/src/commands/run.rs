use std::path::PathBuf;

use clap::Parser;
use eyre::Result;
use openvm_sdk::{
    config::{AppConfig, SdkVmConfig},
    fs::read_exe_from_file,
    Sdk,
};

use crate::util::{read_to_stdin, read_to_struct_toml, Input};

#[derive(Parser)]
#[command(name = "run", about = "Run an OpenVM program")]
pub struct RunCmd {
    #[clap(long, action, help = "Path to OpenVM executable")]
    exe: PathBuf,

    #[clap(long, action, help = "Path to app config TOML file")]
    config: PathBuf,

    #[clap(long, value_parser, help = "Input to OpenVM program")]
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
