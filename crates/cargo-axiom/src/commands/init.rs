use std::path::PathBuf;

use axvm_sdk::{
    config::FullAggConfig,
    fs::{write_agg_pk_to_file, write_evm_verifier_to_file},
    Sdk,
};
use clap::Parser;
use eyre::Result;

#[derive(Parser)]
#[command(
    name = "init",
    about = "Generate default aggregation proving key and SNARK verifier contract"
)]
pub struct InitCmd {
    #[clap(long, action, help = "Path to aggregation proving key output")]
    agg_pk: PathBuf,

    #[clap(long, action, help = "Path to verifier output")]
    verifier: PathBuf,
}

impl InitCmd {
    pub fn run(&self) -> Result<()> {
        let agg_config = FullAggConfig::default();
        let agg_pk = Sdk.agg_keygen(agg_config)?;
        let verifier = Sdk.generate_snark_verifier_contract(&agg_pk)?;
        write_agg_pk_to_file(agg_pk, &self.agg_pk)?;
        write_evm_verifier_to_file(verifier, &self.verifier)?;
        Ok(())
    }
}
