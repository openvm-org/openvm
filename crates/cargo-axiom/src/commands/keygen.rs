use std::path::PathBuf;

use axvm_sdk::{
    config::{AppConfig, FullAggConfig, SdkVmConfig},
    fs::{write_agg_pk_to_file, write_app_pk_to_file},
    Sdk,
};
use clap::Parser;
use eyre::Result;

use crate::util::read_to_struct_toml;

#[derive(Parser)]
#[command(
    name = "keygen",
    about = "Generate an application (or aggregation) proving key"
)]
pub struct KeygenCmd {
    #[clap(long, action, help = "Path to app (or agg) config TOML file")]
    config: PathBuf,

    #[clap(long, action, help = "Path to output file")]
    output: PathBuf,

    #[clap(long, action, help = "Generates aggregation key if present")]
    aggregation: bool,
}

impl KeygenCmd {
    pub fn run(&self) -> Result<()> {
        if self.aggregation {
            let agg_config: FullAggConfig = read_to_struct_toml(&self.config)?;
            let agg_pk = Sdk.agg_keygen(agg_config)?;
            write_agg_pk_to_file(agg_pk, &self.output)?;
        } else {
            let app_config: AppConfig<SdkVmConfig> = read_to_struct_toml(&self.config)?;
            let app_pk = Sdk.app_keygen(app_config)?;
            write_app_pk_to_file(app_pk, &self.output)?;
        }
        Ok(())
    }
}
