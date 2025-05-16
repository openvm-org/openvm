use std::{
    fs::{create_dir_all, write},
    path::PathBuf,
};

use aws_config::{defaults, BehaviorVersion, Region};
use aws_sdk_s3::Client;
use clap::Parser;
use eyre::{eyre, Result};
use openvm_native_recursion::halo2::utils::CacheHalo2ParamsReader;
use openvm_sdk::{
    config::{AggConfig, AggStarkConfig},
    fs::{
        write_agg_halo2_pk_to_file, write_agg_stark_pk_to_file, write_evm_halo2_verifier_to_folder,
        EVM_HALO2_VERIFIER_BASE_NAME, EVM_HALO2_VERIFIER_INTERFACE_NAME,
        EVM_HALO2_VERIFIER_PARENT_NAME,
    },
    DefaultStaticVerifierPvHandler, Sdk,
};

use crate::{
    default::{
        default_agg_halo2_pk_path, default_agg_stark_pk_path, default_asm_path,
        default_evm_halo2_verifier_path, default_params_dir,
    },
    util::read_default_agg_pk,
};

#[derive(Parser)]
#[command(
    name = "setup",
    about = "Set up for generating EVM proofs. ATTENTION: this requires large amounts of computation and memory. "
)]
pub struct SetupCmd {
    #[arg(
        long,
        default_value = "false",
        help = "use --evm to also generate proving keys for EVM verifier"
    )]
    pub evm: bool,
    #[arg(
        long,
        default_value = "false",
        help = "force keygen even if the proving keys already exist"
    )]
    pub force_agg_keygen: bool,
}

impl SetupCmd {
    pub async fn run(&self) -> Result<()> {
        let default_agg_stark_pk_path = default_agg_stark_pk_path();
        let default_params_dir = default_params_dir();
        let default_evm_halo2_verifier_path = default_evm_halo2_verifier_path();
        let default_asm_path = default_asm_path();
        if !self.evm {
            if PathBuf::from(&default_agg_stark_pk_path).exists() {
                println!("Aggregation stark proving key already exists");
                return Ok(());
            }
            let agg_stark_config = AggStarkConfig::default();
            let sdk = Sdk::new();
            let agg_stark_pk = sdk.agg_stark_keygen(agg_stark_config)?;

            println!("Writing stark proving key to file...");
            write_agg_stark_pk_to_file(&agg_stark_pk, default_agg_stark_pk_path)?;

            println!("Generating root verifier ASM...");
            let root_verifier_asm = sdk.generate_root_verifier_asm(&agg_stark_pk);

            println!("Writing root verifier ASM to file...");
            write(&default_asm_path, root_verifier_asm)?;
        } else {
            let default_agg_halo2_pk_path = default_agg_halo2_pk_path();
            if PathBuf::from(&default_agg_stark_pk_path).exists()
                && PathBuf::from(&default_agg_halo2_pk_path).exists()
                && PathBuf::from(&default_evm_halo2_verifier_path)
                    .join(EVM_HALO2_VERIFIER_PARENT_NAME)
                    .exists()
                && PathBuf::from(&default_evm_halo2_verifier_path)
                    .join(EVM_HALO2_VERIFIER_BASE_NAME)
                    .exists()
                && PathBuf::from(&default_evm_halo2_verifier_path)
                    .join("interfaces")
                    .join(EVM_HALO2_VERIFIER_INTERFACE_NAME)
                    .exists()
            {
                println!("Aggregation proving key and verifier contract already exist");
                return Ok(());
            } else if !Self::check_solc_installed() {
                return Err(eyre!(
                    "solc is not installed, please install solc to continue"
                ));
            }

            Self::download_params(10, 24).await?;
            let params_reader = CacheHalo2ParamsReader::new(&default_params_dir);
            let agg_config = AggConfig::default();
            let sdk = Sdk::new();

            let agg_pk = if !self.force_agg_keygen
                && PathBuf::from(&default_agg_stark_pk_path).exists()
                && PathBuf::from(&default_agg_halo2_pk_path).exists()
            {
                read_default_agg_pk()?
            } else {
                println!("Generating proving key...");
                sdk.agg_keygen(agg_config, &params_reader, &DefaultStaticVerifierPvHandler)?
            };

            println!("Generating root verifier ASM...");
            let root_verifier_asm = sdk.generate_root_verifier_asm(&agg_pk.agg_stark_pk);

            println!("Generating verifier contract...");
            let verifier = sdk.generate_halo2_verifier_solidity(&params_reader, &agg_pk)?;

            println!("Writing stark proving key to file...");
            write_agg_stark_pk_to_file(&agg_pk.agg_stark_pk, &default_agg_stark_pk_path)?;

            println!("Writing halo2 proving key to file...");
            write_agg_halo2_pk_to_file(&agg_pk.halo2_pk, &default_agg_halo2_pk_path)?;

            println!("Writing root verifier ASM to file...");
            write(&default_asm_path, root_verifier_asm)?;

            println!("Writing verifier contract to file...");
            write_evm_halo2_verifier_to_folder(verifier, &default_evm_halo2_verifier_path)?;
        }
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
            let file_name = format!("kzg_bn254_{}.srs", k);
            let local_file_path = PathBuf::from(&default_params_dir).join(&file_name);
            if !local_file_path.exists() {
                println!("Downloading {}", file_name);
                let key = format!("challenge_0085/{}", file_name);
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
