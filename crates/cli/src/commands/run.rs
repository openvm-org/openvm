use std::path::PathBuf;

use clap::Parser;
use eyre::Result;
use openvm_sdk::{fs::read_exe_from_file, Sdk};

use crate::{
    default::{DEFAULT_APP_CONFIG_PATH, DEFAULT_APP_EXE_PATH},
    util::{read_config_toml_or_default, read_to_stdin, Input},
};

#[derive(Parser)]
#[command(name = "run", about = "Run an OpenVM program")]
pub struct RunCmd {
    #[clap(long, action, help = "Path to OpenVM executable", default_value = DEFAULT_APP_EXE_PATH)]
    exe: PathBuf,

    #[clap(long, action, help = "Path to app config TOML file", default_value = DEFAULT_APP_CONFIG_PATH)]
    config: PathBuf,

    #[clap(long, value_parser, help = "Input to OpenVM program")]
    input: Option<Input>,
}

impl RunCmd {
    pub fn run(&self) -> Result<()> {
        let exe = read_exe_from_file(&self.exe)?;
        let app_config = read_config_toml_or_default(&self.config)?;
        let output = Sdk.execute(exe, app_config.app_vm_config, read_to_stdin(&self.input)?)?;
        println!("Execution output: {:?}", output);
        Ok(())
    }
}
