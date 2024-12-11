use std::path::PathBuf;

use axvm_sdk::{
    commit::AppExecutionCommit,
    config::SdkVmConfig,
    fs::{
        read_agg_pk_from_file, read_app_pk_from_file, read_exe_from_file, write_app_proof_to_file,
        write_evm_proof_to_file,
    },
    keygen::AppProvingKey,
    Sdk,
};
use clap::Parser;
use eyre::Result;

use crate::util::{read_to_stdin, Input};

#[derive(Parser)]
#[command(name = "prove", about = "Generate a program proof")]
pub struct ProveCmd {
    #[clap(long, action, help = "Path to app proving key")]
    app_pk: PathBuf,

    #[clap(long, action, help = "Path to axVM executable")]
    exe: PathBuf,

    #[clap(long, value_parser, help = "Input to axVM program")]
    input: Option<Input>,

    #[clap(long, action, help = "Path to output proof")]
    output: PathBuf,

    #[clap(
        long,
        action,
        help = "Generates end-to-end EVM proof if present. WARNING: this requires large amounts of computation and memory."
    )]
    evm: bool,

    #[clap(long, action, help = "Path to aggregation proving key")]
    agg_pk: Option<PathBuf>,
}

impl ProveCmd {
    pub fn run(&self) -> Result<()> {
        let app_pk: AppProvingKey<SdkVmConfig> = read_app_pk_from_file(&self.app_pk)?;
        let app_exe = read_exe_from_file(&self.exe)?;
        let committed_exe = Sdk.commit_app_exe(app_pk.app_fri_params(), app_exe)?;

        let commits = AppExecutionCommit::compute(
            &app_pk.app_vm_pk.vm_config,
            &committed_exe,
            &app_pk.leaf_committed_exe,
        );
        println!("app_pk commit: {:?}", commits.app_config_commit_to_bn254());
        println!("exe commit: {:?}", commits.exe_commit_to_bn254());

        let input = read_to_stdin(&self.input)?;
        if self.evm {
            println!("Generating EVM proof, this may take a lot of compute and memory...");
            let agg_pk = read_agg_pk_from_file(self.agg_pk.as_ref().unwrap())?;
            let evm_proof = Sdk.generate_evm_proof(app_pk, committed_exe, agg_pk, input)?;
            write_evm_proof_to_file(evm_proof, &self.output)?;
        } else {
            let app_proof = Sdk.generate_app_proof(app_pk, committed_exe, input)?;
            write_app_proof_to_file(app_proof, &self.output)?;
        }
        Ok(())
    }
}
