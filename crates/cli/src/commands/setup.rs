use std::{
    fs::{create_dir_all, write},
    path::PathBuf,
};

use aws_config::{defaults, BehaviorVersion, Region};
use aws_sdk_s3::Client;
use clap::Parser;
use eyre::{eyre, Result};
use openvm_sdk::{
    config::AggregationSystemParams,
    fs::{
        write_evm_halo2_verifier_to_folder, EVM_HALO2_VERIFIER_BASE_NAME,
        EVM_HALO2_VERIFIER_INTERFACE_NAME, EVM_HALO2_VERIFIER_PARENT_NAME,
    },
    Sdk, OPENVM_VERSION,
};
use openvm_stark_sdk::config::{app_params_with_100_bits_security, MAX_APP_LOG_STACKED_HEIGHT};

use crate::default::{default_evm_halo2_verifier_path, default_params_dir};

/// The maximum value of `k` to download Halo2 KZG trusted setup parameters for. This depends on the
/// default verifier circuit and wrapper circuit sizes.
const MAX_HALO2_VERIFIER_K_FOR_DOWNLOAD: usize = 24;

#[derive(Parser)]
#[command(
    name = "setup",
    about = "Set up for generating EVM proofs. ATTENTION: this requires large amounts of computation and memory. "
)]
pub struct SetupCmd {
    #[arg(
        long,
        default_value = "false",
        help = "Force verifier regeneration even if the verifier contract already exists"
    )]
    pub force: bool,
}

impl SetupCmd {
    pub async fn run(&self) -> Result<()> {
        if !Self::check_solc_installed() {
            return Err(eyre!(
                "solc is not installed, please install solc to continue"
            ));
        }

        Self::download_params(10, MAX_HALO2_VERIFIER_K_FOR_DOWNLOAD as u32).await?;

        let verifier_dir = PathBuf::from(default_evm_halo2_verifier_path());
        let versioned_verifier_dir = verifier_dir.join("src").join(format!("v{OPENVM_VERSION}"));
        let interface_dir = versioned_verifier_dir.join("interfaces");

        if !self.force
            && versioned_verifier_dir
                .join(EVM_HALO2_VERIFIER_PARENT_NAME)
                .exists()
            && versioned_verifier_dir
                .join(EVM_HALO2_VERIFIER_BASE_NAME)
                .exists()
            && interface_dir
                .join(EVM_HALO2_VERIFIER_INTERFACE_NAME)
                .exists()
        {
            println!(
                "EVM verifier artifacts already exist in {}",
                verifier_dir.display()
            );
            return Ok(());
        }

        let sdk = Sdk::standard(
            app_params_with_100_bits_security(MAX_APP_LOG_STACKED_HEIGHT),
            AggregationSystemParams::default(),
        );

        println!("Generating verifier contract...");
        let verifier = sdk.generate_halo2_verifier_solidity()?;

        println!("Writing verifier contract to {}", verifier_dir.display());
        write_evm_halo2_verifier_to_folder(verifier, &verifier_dir)?;
        Ok(())
    }

    fn check_solc_installed() -> bool {
        std::process::Command::new("solc")
            .arg("--version")
            .output()
            .is_ok()
    }

    async fn download_params(min_k: u32, max_k: u32) -> Result<()> {
        let default_params_dir = default_params_dir();
        create_dir_all(&default_params_dir)?;

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
                    .await?;
                let data = resp.body.collect().await?;
                write(local_file_path, data.into_bytes())?;
            }
        }

        Ok(())
    }
}
