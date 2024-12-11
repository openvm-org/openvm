use std::path::PathBuf;

use axvm_sdk::{
    config::SdkVmConfig,
    fs::{
        read_agg_pk_from_file, read_app_pk_from_file, read_app_proof_from_file,
        read_evm_proof_from_file, read_evm_verifier_from_file,
    },
    keygen::AppProvingKey,
    Sdk,
};
use clap::Parser;
use eyre::{eyre, Result};

#[derive(Parser)]
#[command(name = "verify", about = "Verify a proof")]
pub struct VerifyCmd {
    #[clap(long, action, help = "Path to app (or agg) proving key")]
    pk: Option<PathBuf>,

    #[clap(long, action, help = "Path to app (or evm) proof")]
    proof: PathBuf,

    #[clap(long, action, help = "Verifies end-to-end EVM proof if present")]
    evm: bool,

    #[clap(long, action, help = "Path to EVM verifier")]
    evm_verifier: Option<PathBuf>,
}

impl VerifyCmd {
    pub fn run(&self) -> Result<()> {
        if self.evm {
            let evm_verifier = if let Some(path) = &self.evm_verifier {
                read_evm_verifier_from_file(path)?
            } else {
                let agg_pk = read_agg_pk_from_file(self.pk.as_ref().unwrap())?;
                Sdk.generate_snark_verifier_contract(&agg_pk)?
            };
            let evm_proof = read_evm_proof_from_file(&self.proof)?;
            if !Sdk.verify_evm_proof(&evm_verifier, &evm_proof) {
                return Err(eyre!("EVM proof verification failed"));
            }
        } else {
            let app_pk: AppProvingKey<SdkVmConfig> =
                read_app_pk_from_file(self.pk.as_ref().unwrap())?;
            let app_proof = read_app_proof_from_file(&self.proof)?;
            Sdk.verify_app_proof(&app_pk, &app_proof)?;
        }
        Ok(())
    }
}
