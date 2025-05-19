use std::{
    fs::{copy, create_dir_all},
    path::{Path, PathBuf},
};

use clap::Parser;
use eyre::Result;
use openvm_sdk::{
    fs::{write_app_pk_to_file, write_app_vk_to_file},
    Sdk,
};

use crate::{
    default::{DEFAULT_APP_CONFIG_PATH, DEFAULT_APP_PK_NAME, DEFAULT_APP_VK_NAME},
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
        help = "Output directory that OpenVM proving artifacts will be copied to",
        help_heading = "OpenVM Options"
    )]
    output_dir: Option<PathBuf>,

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
        let app_pk_path = app_pk_path(&target_dir);
        let app_vk_path = app_vk_path(&target_dir);

        keygen(
            &self.config,
            &app_pk_path,
            &app_vk_path,
            self.output_dir.as_ref(),
        )?;
        println!(
            "Successfully generated app pk and vk in {}",
            if let Some(output_dir) = self.output_dir.as_ref() {
                output_dir.display()
            } else {
                app_pk_path.parent().unwrap().display()
            }
        );
        Ok(())
    }
}

pub(crate) fn keygen(
    config: impl AsRef<Path>,
    app_pk_path: impl AsRef<Path>,
    app_vk_path: impl AsRef<Path>,
    output_dir: Option<impl AsRef<Path>>,
) -> Result<()> {
    let app_config = read_config_toml_or_default(config)?;
    let app_pk = Sdk::new().app_keygen(app_config)?;
    let app_vk = app_pk.get_app_vk();
    write_app_vk_to_file(app_vk, &app_vk_path)?;
    write_app_pk_to_file(app_pk, &app_pk_path)?;

    if let Some(output_dir) = output_dir {
        let output_dir = output_dir.as_ref();
        create_dir_all(output_dir)?;
        copy(&app_pk_path, output_dir.join(DEFAULT_APP_PK_NAME))?;
        copy(&app_vk_path, output_dir.join(DEFAULT_APP_VK_NAME))?;
    }

    Ok(())
}
