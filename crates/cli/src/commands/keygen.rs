use std::{
    fs::{copy, create_dir_all},
    path::{Path, PathBuf},
};

use clap::Parser;
use eyre::{Context, Result};
use openvm_sdk::{config::AggregationSystemParams, fs::write_object_to_file, Sdk};

use crate::{
    args::ManifestArgs,
    default::{
        default_app_config, DEFAULT_AGG_PREFIX_PK_NAME, DEFAULT_APP_PK_NAME, DEFAULT_APP_VK_NAME,
        OPENVM_CONFIG_FILENAME,
    },
    util::{
        get_agg_prefix_pk_path, get_app_pk_path, get_app_vk_path, get_manifest_path_and_dir,
        get_target_dir, read_config_toml_or_default,
    },
};

#[derive(Parser)]
#[command(name = "keygen", about = "Generate application and aggregation keys")]
pub struct KeygenCmd {
    #[arg(
        long,
        help = "Path to the OpenVM config .toml file that specifies the VM extensions, by default will search for the file at ${manifest_dir}/openvm.toml",
        help_heading = "OpenVM Options"
    )]
    config: Option<PathBuf>,

    #[arg(
        long,
        help = "Output directory that OpenVM proving artifacts will be copied to",
        help_heading = "OpenVM Options"
    )]
    output_dir: Option<PathBuf>,

    #[arg(
        long,
        help = "Only generate app keys (app.pk/app.vk), skip the aggregation prefix proving key (agg_prefix.pk)",
        help_heading = "OpenVM Options"
    )]
    app_only: bool,

    #[command(flatten)]
    cargo_args: KeygenCargoArgs,
}

#[derive(Parser)]
pub struct KeygenCargoArgs {
    #[clap(flatten)]
    pub(crate) manifest: ManifestArgs,
}

impl KeygenCmd {
    pub fn run(&self) -> Result<()> {
        let (manifest_path, manifest_dir) =
            get_manifest_path_and_dir(&self.cargo_args.manifest.manifest_path)?;
        let target_dir = get_target_dir(&self.cargo_args.manifest.target_dir, &manifest_path);
        let app_pk_path = get_app_pk_path(&target_dir);
        let app_vk_path = get_app_vk_path(&target_dir);
        let agg_prefix_pk_path = get_agg_prefix_pk_path(&target_dir);

        keygen(
            self.config
                .to_owned()
                .unwrap_or_else(|| manifest_dir.join(OPENVM_CONFIG_FILENAME)),
            &app_pk_path,
            &app_vk_path,
            &agg_prefix_pk_path,
            self.output_dir.as_ref(),
            !self.app_only,
        )?;
        println!(
            "Successfully generated {} in {}",
            if self.app_only {
                "app pk and vk"
            } else {
                "app pk/vk and agg_prefix pk"
            },
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
    agg_prefix_pk_path: impl AsRef<Path>,
    output_dir: Option<impl AsRef<Path>>,
    generate_agg: bool,
) -> Result<()> {
    let app_config = read_config_toml_or_default(config)?;
    assert_default_root_shape(&app_config);
    let sdk = Sdk::new(app_config, AggregationSystemParams::default())?;
    let (app_pk, app_vk) = sdk.app_keygen();
    write_object_to_file(&app_vk_path, app_vk)?;
    write_object_to_file(&app_pk_path, app_pk)?;
    if generate_agg {
        write_object_to_file(&agg_prefix_pk_path, sdk.agg_prefix_pk())?;
    }

    if let Some(output_dir) = output_dir {
        let output_dir = output_dir.as_ref();
        create_dir_all(output_dir)
            .with_context(|| format!("failed to create directory {}", output_dir.display()))?;
        copy(&app_pk_path, output_dir.join(DEFAULT_APP_PK_NAME))
            .with_context(|| format!("failed to copy app pk to {}", output_dir.display()))?;
        copy(&app_vk_path, output_dir.join(DEFAULT_APP_VK_NAME))
            .with_context(|| format!("failed to copy app vk to {}", output_dir.display()))?;
        if generate_agg {
            copy(
                &agg_prefix_pk_path,
                output_dir.join(DEFAULT_AGG_PREFIX_PK_NAME),
            )
            .with_context(|| format!("failed to copy agg prefix pk to {}", output_dir.display()))?;
        }
    }

    Ok(())
}

fn assert_default_root_shape(
    app_config: &openvm_sdk::config::AppConfig<openvm_sdk_config::SdkVmConfig>,
) {
    let default_system_config = default_app_config().app_vm_config;
    let actual_system_config = app_config.app_vm_config.as_ref();
    let default_system_config = default_system_config.as_ref();

    assert_eq!(
        actual_system_config.num_public_values, default_system_config.num_public_values,
        "cargo openvm keygen only supports the default num_public_values"
    );
    let actual_memory_dimensions = actual_system_config.memory_config.memory_dimensions();
    let default_memory_dimensions = default_system_config.memory_config.memory_dimensions();
    assert!(
        actual_memory_dimensions.addr_space_height == default_memory_dimensions.addr_space_height
            && actual_memory_dimensions.address_height == default_memory_dimensions.address_height,
        "cargo openvm keygen only supports the default memory_dimensions"
    );
}
