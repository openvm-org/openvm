use std::path::PathBuf;

use axvm_sdk::{
    config::{AppConfig, SdkVmConfig},
    fs::write_app_pk_to_file,
    Sdk,
};
use clap::Parser;
use eyre::Result;

use crate::util::read_to_struct_toml;

#[derive(Parser)]
#[command(name = "keygen", about = "Generate an application proving key")]
pub struct KeygenCmd {
    #[clap(long, action, help = "Path to app config TOML file")]
    config: PathBuf,

    #[clap(long, action, help = "Path to output file")]
    output: PathBuf,
}

impl KeygenCmd {
    pub fn run(&self) -> Result<()> {
        let app_config: AppConfig<SdkVmConfig> = read_to_struct_toml(&self.config)?;
        let app_pk = Sdk.app_keygen(app_config)?;
        write_app_pk_to_file(app_pk, &self.output)?;
        Ok(())
    }
}
