use std::path::PathBuf;

use axvm_native_recursion::halo2::utils::CacheHalo2ParamsReader;
use axvm_sdk::{
    config::AggConfig,
    fs::{write_agg_pk_to_file, write_evm_verifier_to_file},
    Sdk,
};
use clap::Parser;
use eyre::Result;

pub const AGG_PK_PATH: &str = "~/.axvm/agg.pk";
pub const VERIFIER_PATH: &str = "~/.axvm/verifier.sol";

#[derive(Parser)]
#[command(
    name = "evm-proving-setup",
    about = "Set up for generating EVM proofs. ATTENTION: this requires large amounts of computation and memory. "
)]
pub struct EvmProvingSetupCmd {}

impl EvmProvingSetupCmd {
    pub fn run(&self) -> Result<()> {
        if PathBuf::from(AGG_PK_PATH).exists() && PathBuf::from(VERIFIER_PATH).exists() {
            println!("Aggregation proving key and verifier contract already exist");
            return Ok(());
        }
        // FIXME: read path from config.
        let params_reader = CacheHalo2ParamsReader::new_with_default_params_dir();
        let agg_config = AggConfig::default();
        let agg_pk = Sdk.agg_keygen(agg_config, &params_reader)?;
        let verifier = Sdk.generate_snark_verifier_contract(&params_reader, &agg_pk)?;
        write_agg_pk_to_file(agg_pk, AGG_PK_PATH)?;
        write_evm_verifier_to_file(verifier, VERIFIER_PATH)?;
        Ok(())
    }
}
