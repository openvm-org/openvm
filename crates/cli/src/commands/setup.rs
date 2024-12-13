use std::path::PathBuf;

use clap::Parser;
use eyre::{eyre, Result};
use openvm_native_recursion::halo2::utils::CacheHalo2ParamsReader;
use openvm_sdk::{
    config::AggConfig,
    fs::{write_agg_pk_to_file, write_evm_verifier_to_file},
    Sdk,
};

use crate::default::{DEFAULT_AGG_PK_PATH, DEFAULT_VERIFIER_PATH};

#[derive(Parser)]
#[command(
    name = "evm-proving-setup",
    about = "Set up for generating EVM proofs. ATTENTION: this requires large amounts of computation and memory. "
)]
pub struct EvmProvingSetupCmd {}

impl EvmProvingSetupCmd {
    pub fn run(&self) -> Result<()> {
        if PathBuf::from(DEFAULT_AGG_PK_PATH).exists()
            && PathBuf::from(DEFAULT_VERIFIER_PATH).exists()
        {
            println!("Aggregation proving key and verifier contract already exist");
            return Ok(());
        } else if !Self::check_solc_installed() {
            return Err(eyre!(
                "solc is not installed, please install solc to continue"
            ));
        }

        // FIXME: read path from config.
        let params_reader = CacheHalo2ParamsReader::new_with_default_params_dir();
        let agg_config = AggConfig::default();
        let agg_pk = Sdk.agg_keygen(agg_config, &params_reader)?;
        let verifier = Sdk.generate_snark_verifier_contract(&params_reader, &agg_pk)?;
        write_agg_pk_to_file(agg_pk, DEFAULT_AGG_PK_PATH)?;
        write_evm_verifier_to_file(verifier, DEFAULT_VERIFIER_PATH)?;
        Ok(())
    }

    fn check_solc_installed() -> bool {
        std::process::Command::new("solc")
            .arg("--version")
            .output()
            .is_ok()
    }
}
