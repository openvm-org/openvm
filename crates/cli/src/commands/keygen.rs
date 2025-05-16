use std::path::{Path, PathBuf};

use clap::Parser;
use eyre::Result;
use openvm_sdk::{
    fs::{write_app_pk_to_file, write_app_vk_to_file},
    Sdk,
};

use crate::{
    default::DEFAULT_APP_CONFIG_PATH,
    global::{app_pk_path, app_vk_path, manifest_path_and_dir, target_dir},
    util::read_config_toml_or_default,
};

#[derive(Parser)]
#[command(name = "keygen", about = "Generate an application proving key")]
pub struct KeygenCmd {
    #[arg(
        long,
        default_value = DEFAULT_APP_CONFIG_PATH,
        help = "Path to the OpenVM config .toml file that specifies the VM extensions",
        help_heading = "OpenVM Options"
    )]
    config: PathBuf,

    #[arg(
        long,
        help = "Output path for the app proving key, by default will be ${target_dir}/openvm/app.pk",
        help_heading = "OpenVM Options"
    )]
    app_pk: Option<PathBuf>,

    #[arg(
        long,
        help = "Output path for the app verifying key, by default will be ${target_dir}/openvm/app.vk",
        help_heading = "OpenVM Options"
    )]
    app_vk: Option<PathBuf>,

    #[command(flatten)]
    cargo_args: KeygenCargoArgs,
}

#[derive(Parser)]
pub struct KeygenCargoArgs {
    #[arg(
        long,
        value_name = "DIR",
        help = "Directory for all Cargo-generated artifacts and intermediate files",
        help_heading = "Cargo Options"
    )]
    pub(crate) target_dir: Option<PathBuf>,

    #[arg(
        long,
        value_name = "PATH",
        help = "Path to the Cargo.toml file, by default searches for the file in the current or any parent directory",
        help_heading = "Cargo Options"
    )]
    pub(crate) manifest_path: Option<PathBuf>,
}

impl KeygenCmd {
    pub fn run(&self) -> Result<()> {
        let (manifest_path, _) = manifest_path_and_dir(&self.cargo_args.manifest_path)?;
        let target_dir = target_dir(&self.cargo_args.target_dir, &manifest_path);
        let app_pk_path = app_pk_path(&self.app_pk, &target_dir);
        let app_vk_path = app_vk_path(&self.app_vk, &target_dir);

        keygen(&self.config, &app_pk_path, &app_vk_path)?;
        println!(
            "Successfully generated app proving key and vk in {}",
            app_pk_path.display()
        );
        Ok(())
    }
}

pub(crate) fn keygen(
    config: impl AsRef<Path>,
    app_pk_path: impl AsRef<Path>,
    app_vk_path: impl AsRef<Path>,
) -> Result<()> {
    let app_config = read_config_toml_or_default(config)?;
    let app_pk = Sdk::new().app_keygen(app_config)?;
    let app_vk = app_pk.get_app_vk();
    write_app_vk_to_file(app_vk, app_vk_path)?;
    write_app_pk_to_file(app_pk, app_pk_path)?;
    Ok(())
}
