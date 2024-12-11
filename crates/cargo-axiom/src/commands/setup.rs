use std::path::PathBuf;

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
    about = "Generate default aggregation proving key and SNARK verifier contract"
)]
pub struct EvmProvingSetupCmd {}

impl EvmProvingSetupCmd {
    pub fn run(&self) -> Result<()> {
        if PathBuf::from(AGG_PK_PATH).exists() && PathBuf::from(VERIFIER_PATH).exists() {
            println!("Aggregation proving key and verifier contract already exist");
            return Ok(());
        }
        let agg_config = AggConfig::default();
        let agg_pk = Sdk.agg_keygen(agg_config)?;
        let verifier = Sdk.generate_snark_verifier_contract(&agg_pk)?;
        write_agg_pk_to_file(agg_pk, AGG_PK_PATH)?;
        write_evm_verifier_to_file(verifier, VERIFIER_PATH)?;
        Ok(())
    }
}
