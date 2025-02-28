use std::path::PathBuf;

use clap::Parser;
use eyre::Result;
use openvm_circuit::arch::instructions::exe::VmExe;
use openvm_sdk::{
    config::{AppConfig, SdkVmConfig},
    fs::read_exe_from_file,
    Sdk, F,
};

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
        execute(self.exe.clone(), &self.config, &self.input)?;
        Ok(())
    }
}

pub(crate) fn execute(
    exe_path: PathBuf,
    config: &PathBuf,
    input: &Option<Input>,
) -> Result<(VmExe<F>, AppConfig<SdkVmConfig>)> {
    let exe = read_exe_from_file(exe_path)?;
    let app_config = read_config_toml_or_default(config)?;
    let output = Sdk.execute(
        exe.clone(),
        app_config.app_vm_config.clone(),
        read_to_stdin(input)?,
    )?;
    println!("Execution output: {:?}", output);
    Ok((exe, app_config))
}
