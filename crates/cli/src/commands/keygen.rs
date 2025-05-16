use std::path::{Path, PathBuf};

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
    #[arg(long, action, help = "Path to app config TOML file", default_value = DEFAULT_APP_CONFIG_PATH)]
    config: PathBuf,

    #[arg(
        long,
        action,
        help = "Path to output app proving key file",
        default_value = DEFAULT_APP_PK_PATH
    )]
    output: PathBuf,

    #[arg(
        long,
        action,
        help = "Path to output app verifying key file",
        default_value = DEFAULT_APP_VK_PATH
    )]
    vk_output: PathBuf,
}

impl KeygenCmd {
    pub fn run(&self) -> Result<()> {
        keygen(&self.config, &self.output, &self.vk_output)?;
        Ok(())
    }
}

pub(crate) fn keygen(
    config: impl AsRef<Path>,
    output: impl AsRef<Path>,
    vk_output: impl AsRef<Path>,
) -> Result<()> {
    let app_config = read_config_toml_or_default(config)?;
    let app_pk = Sdk::new().app_keygen(app_config)?;
    write_app_vk_to_file(app_pk.get_app_vk(), vk_output.as_ref())?;
    write_app_pk_to_file(app_pk, output.as_ref())?;
    Ok(())
}
