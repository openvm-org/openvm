use std::{
    fs::{create_dir_all, write},
    path::{Path, PathBuf},
};

use aws_config::{defaults, BehaviorVersion, Region};
use aws_sdk_s3::Client;
use clap::Parser;
use eyre::{eyre, Context, Result};
use openvm_sdk::{
    config::AggregationSystemParams,
    fs::{
        write_object_to_file, EVM_HALO2_VERIFIER_BASE_NAME, EVM_HALO2_VERIFIER_INTERFACE_NAME,
        EVM_HALO2_VERIFIER_PARENT_NAME, EVM_VERIFIER_ARTIFACT_FILENAME,
    },
    Sdk, OPENVM_VERSION,
};

use crate::default::{
    default_app_config, default_evm_halo2_verifier_path, default_internal_recursive_pk_path,
    default_params_dir, default_root_pk_path,
};

/// The maximum value of `k` to download Halo2 KZG trusted setup parameters for. This depends on the
/// default verifier circuit and wrapper circuit sizes.
const MAX_HALO2_VERIFIER_K_FOR_DOWNLOAD: usize = 24;

#[derive(Parser)]
#[command(name = "setup", about = "Set up OpenVM recursive proving artifacts.")]
pub struct SetupCmd {
    #[arg(
        long,
        default_value = "false",
        help = "Force verifier regeneration even if the verifier artifacts already exist"
    )]
    pub force: bool,

    #[arg(
        long,
        default_value = "false",
        help = "Also cache the root proving key and download EVM verifier artifacts. ATTENTION: this requires large amounts of computation and memory."
    )]
    pub evm: bool,

    #[arg(
        long,
        default_value = "false",
        help = "Download pre-built EVM verifier artifacts from S3 instead of generating locally"
    )]
    pub download: bool,
}

impl SetupCmd {
    pub async fn run(&self) -> Result<()> {
        let sdk = Sdk::new(default_app_config(), AggregationSystemParams::default())?;

        let internal_recursive_pk_path = PathBuf::from(default_internal_recursive_pk_path());
        println!(
            "Writing internal-recursive proving key to {}",
            internal_recursive_pk_path.display()
        );
        write_object_to_file(&internal_recursive_pk_path, sdk.agg_pk().internal_recursive)?;

        if !self.evm {
            return Ok(());
        }

        let root_pk_path = PathBuf::from(default_root_pk_path());
        println!("Writing root proving key to {}", root_pk_path.display());
        write_object_to_file(&root_pk_path, sdk.root_pk())?;

        Self::download_params(10, MAX_HALO2_VERIFIER_K_FOR_DOWNLOAD as u32).await?;

        let verifier_dir = PathBuf::from(default_evm_halo2_verifier_path());
        let versioned_verifier_dir = verifier_dir.join("src").join(format!("v{OPENVM_VERSION}"));
        if !self.force && Self::verifier_artifacts_exist(&versioned_verifier_dir) {
            println!(
                "EVM verifier artifacts already exist in {}",
                verifier_dir.display()
            );
            return Ok(());
        }

        if self.download {
            Self::download_verifier(&versioned_verifier_dir).await?;
        } else {
            Self::generate_verifier(&sdk, &verifier_dir)?;
        }

        Ok(())
    }

    fn verifier_artifacts_exist(versioned_verifier_dir: &Path) -> bool {
        versioned_verifier_dir
            .join(EVM_HALO2_VERIFIER_PARENT_NAME)
            .exists()
            && versioned_verifier_dir
                .join(EVM_HALO2_VERIFIER_BASE_NAME)
                .exists()
            && versioned_verifier_dir
                .join(EVM_VERIFIER_ARTIFACT_FILENAME)
                .exists()
            && versioned_verifier_dir
                .join("interfaces")
                .join(EVM_HALO2_VERIFIER_INTERFACE_NAME)
                .exists()
    }

    fn check_solc_installed() -> bool {
        std::process::Command::new("solc")
            .arg("--version")
            .output()
            .is_ok()
    }

    fn generate_verifier(sdk: &Sdk, verifier_dir: &Path) -> Result<()> {
        if !Self::check_solc_installed() {
            return Err(eyre!(
                "solc is not installed, please install solc or rerun with --download"
            ));
        }

        #[cfg(feature = "evm-verify")]
        {
            use openvm_sdk::fs::write_evm_halo2_verifier_to_folder;

            println!("Generating verifier contract locally.");
            let verifier = sdk.generate_halo2_verifier_solidity()?;
            println!("Writing verifier contract to {}", verifier_dir.display());
            write_evm_halo2_verifier_to_folder(verifier, verifier_dir)?;
            Ok(())
        }

        #[cfg(not(feature = "evm-verify"))]
        {
            let _ = sdk;
            let _ = verifier_dir;
            Err(eyre!(
                "this cargo-openvm build does not include local EVM verifier generation support; rerun with --download"
            ))
        }
    }

    async fn download_verifier(versioned_verifier_dir: &PathBuf) -> Result<()> {
        create_dir_all(versioned_verifier_dir)?;
        let interface_dir = versioned_verifier_dir.join("interfaces");
        create_dir_all(&interface_dir)?;

        let config = defaults(BehaviorVersion::latest())
            .region(Region::new("us-east-1"))
            .no_credentials()
            .load()
            .await;
        let client = Client::new(&config);

        const ARTIFACTS_BUCKET: &str = "openvm-public-artifacts-us-east-1";
        const FULL_VERSION: &str = env!("CARGO_PKG_VERSION");

        let halo2_s3_prefix = format!("v{FULL_VERSION}/halo2/src/v{OPENVM_VERSION}");
        let files = [
            (
                EVM_HALO2_VERIFIER_PARENT_NAME,
                versioned_verifier_dir.join(EVM_HALO2_VERIFIER_PARENT_NAME),
            ),
            (
                EVM_HALO2_VERIFIER_BASE_NAME,
                versioned_verifier_dir.join(EVM_HALO2_VERIFIER_BASE_NAME),
            ),
            (
                EVM_VERIFIER_ARTIFACT_FILENAME,
                versioned_verifier_dir.join(EVM_VERIFIER_ARTIFACT_FILENAME),
            ),
            (
                EVM_HALO2_VERIFIER_INTERFACE_NAME,
                interface_dir.join(EVM_HALO2_VERIFIER_INTERFACE_NAME),
            ),
        ];

        for (name, local_path) in &files {
            if !local_path.exists() {
                let key = if *name == EVM_HALO2_VERIFIER_INTERFACE_NAME {
                    format!("{halo2_s3_prefix}/interfaces/{name}")
                } else {
                    format!("{halo2_s3_prefix}/{name}")
                };
                println!("Downloading {name}");
                let resp = client
                    .get_object()
                    .bucket(ARTIFACTS_BUCKET)
                    .key(&key)
                    .send()
                    .await
                    .map_err(|e| eyre!("Failed to download s3://{ARTIFACTS_BUCKET}/{key}: {e}"))?;
                let data = resp.body.collect().await?;
                write(local_path, data.into_bytes())?;
            }
        }

        Ok(())
    }

    async fn download_params(min_k: u32, max_k: u32) -> Result<()> {
        let default_params_dir = default_params_dir();
        create_dir_all(&default_params_dir)
            .with_context(|| format!("failed to create params directory {default_params_dir}"))?;

        let config = defaults(BehaviorVersion::latest())
            .region(Region::new("us-east-1"))
            .no_credentials()
            .load()
            .await;
        let client = Client::new(&config);

        for k in min_k..=max_k {
            let file_name = format!("kzg_bn254_{k}.srs");
            let local_file_path = PathBuf::from(&default_params_dir).join(&file_name);
            if !local_file_path.exists() {
                println!("Downloading {file_name}");
                let key = format!("challenge_0085/{file_name}");
                let resp = client
                    .get_object()
                    .bucket("axiom-crypto")
                    .key(&key)
                    .send()
                    .await
                    .with_context(|| format!("failed to download {file_name} from S3"))?;
                let data = resp
                    .body
                    .collect()
                    .await
                    .with_context(|| format!("failed to download {file_name} body from S3"))?;
                write(&local_file_path, data.into_bytes())
                    .with_context(|| format!("failed to write {}", local_file_path.display()))?;
            }
        }

        Ok(())
    }
}
