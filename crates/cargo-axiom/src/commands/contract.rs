use std::path::PathBuf;

use axvm_sdk::{
    fs::{read_agg_pk_from_file, write_evm_verifier_to_file},
    Sdk,
};
use clap::Parser;
use eyre::Result;

#[derive(Parser)]
#[command(name = "contract", about = "Generate final SNARK verifier contract")]
pub struct ContractCmd {
    #[clap(long, action, help = "Path to aggregation proving key")]
    agg_pk: PathBuf,

    #[clap(long, action, help = "Path to output file")]
    output: PathBuf,
}

impl ContractCmd {
    pub fn run(&self) -> Result<()> {
        let agg_pk = read_agg_pk_from_file(&self.agg_pk)?;
        let verifier = Sdk.generate_snark_verifier_contract(&agg_pk)?;
        write_evm_verifier_to_file(verifier, &self.output)?;
        Ok(())
    }
}
