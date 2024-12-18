use std::path::PathBuf;

use clap::Parser;
use eyre::Result;
use openvm_sdk::{
    fs::{write_app_pk_to_file, write_app_vk_to_file},
    Sdk,
};

use crate::{
    default::{DEFAULT_APP_CONFIG_PATH, DEFAULT_APP_PK_PATH, DEFAULT_APP_VK_PATH},
    util::read_config_toml_or_default,
};

#[derive(Parser)]
#[command(name = "keygen", about = "Generate an application proving key")]
pub struct KeygenCmd {
    #[clap(long, action, help = "Path to app config TOML file", default_value = DEFAULT_APP_CONFIG_PATH)]
    config: PathBuf,

    #[clap(
        long,
        action,
        help = "Path to output app proving key file",
        default_value = DEFAULT_APP_PK_PATH
    )]
    output: PathBuf,

    #[clap(
        long,
        action,
        help = "Path to output app verifying key file",
        default_value = DEFAULT_APP_VK_PATH
    )]
    vk_output: PathBuf,
}

impl KeygenCmd {
    pub fn run(&self) -> Result<()> {
        let app_config = read_config_toml_or_default(&self.config)?;
        let app_pk = Sdk.app_keygen(app_config)?;
        write_app_vk_to_file(app_pk.get_vk(), &self.vk_output)?;
        write_app_pk_to_file(app_pk, &self.output)?;
        Ok(())
    }
}
