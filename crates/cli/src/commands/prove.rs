use std::{path::PathBuf, sync::Arc};

use clap::Parser;
use eyre::Result;
use openvm_native_recursion::halo2::utils::CacheHalo2ParamsReader;
use openvm_sdk::{
    commit::AppExecutionCommit,
    config::SdkVmConfig,
    fs::{
        read_agg_pk_from_file, read_app_pk_from_file, read_exe_from_file, write_app_proof_to_file,
        write_evm_proof_to_file,
    },
    keygen::AppProvingKey,
    NonRootCommittedExe, Sdk, StdIn,
};

use crate::{
    commands::AGG_PK_PATH,
    util::{read_to_stdin, Input},
};

#[derive(Parser)]
#[command(name = "prove", about = "Generate a program proof")]
pub struct ProveCmd {
    #[clap(subcommand)]
    command: ProveSubCommand,
}

#[derive(Parser)]
enum ProveSubCommand {
    App {
        #[clap(long, action, help = "Path to app proving key")]
        app_pk: PathBuf,

        #[clap(long, action, help = "Path to OpenVM executable")]
        exe: PathBuf,

        #[clap(long, value_parser, help = "Input to OpenVM program")]
        input: Option<Input>,

        #[clap(long, action, help = "Path to output proof")]
        output: PathBuf,
    },
    Evm {
        #[clap(long, action, help = "Path to app proving key")]
        app_pk: PathBuf,

        #[clap(long, action, help = "Path to OpenVM executable")]
        exe: PathBuf,

        #[clap(long, value_parser, help = "Input to OpenVM program")]
        input: Option<Input>,

        #[clap(long, action, help = "Path to output proof")]
        output: PathBuf,
    },
}

impl ProveCmd {
    pub fn run(&self) -> Result<()> {
        match &self.command {
            ProveSubCommand::App {
                app_pk,
                exe,
                input,
                output,
            } => {
                let (app_pk, committed_exe, input) = Self::prepare_execution(app_pk, exe, input)?;
                let app_proof = Sdk.generate_app_proof(app_pk, committed_exe, input)?;
                write_app_proof_to_file(app_proof, output)?;
            }
            ProveSubCommand::Evm {
                app_pk,
                exe,
                input,
                output,
            } => {
                // FIXME: read path from config.
                let params_reader = CacheHalo2ParamsReader::new_with_default_params_dir();
                let (app_pk, committed_exe, input) = Self::prepare_execution(app_pk, exe, input)?;
                println!("Generating EVM proof, this may take a lot of compute and memory...");
                let agg_pk = read_agg_pk_from_file(AGG_PK_PATH).map_err(|e| {
                    eyre::eyre!("Failed to read aggregation proving key: {}\nPlease run 'cargo openvm evm-proving-setup' first", e)
                })?;
                let evm_proof =
                    Sdk.generate_evm_proof(&params_reader, app_pk, committed_exe, agg_pk, input)?;
                write_evm_proof_to_file(evm_proof, output)?;
            }
        }
        Ok(())
    }

    fn prepare_execution(
        app_pk: &PathBuf,
        exe: &PathBuf,
        input: &Option<Input>,
    ) -> Result<(
        Arc<AppProvingKey<SdkVmConfig>>,
        Arc<NonRootCommittedExe>,
        StdIn,
    )> {
        let app_pk: Arc<AppProvingKey<SdkVmConfig>> = Arc::new(read_app_pk_from_file(app_pk)?);
        let app_exe = read_exe_from_file(exe)?;
        let committed_exe = Sdk.commit_app_exe(app_pk.app_fri_params(), app_exe)?;

        let commits = AppExecutionCommit::compute(
            &app_pk.app_vm_pk.vm_config,
            &committed_exe,
            &app_pk.leaf_committed_exe,
        );
        println!("app_pk commit: {:?}", commits.app_config_commit_to_bn254());
        println!("exe commit: {:?}", commits.exe_commit_to_bn254());

        let input = read_to_stdin(input)?;
        Ok((app_pk, committed_exe, input))
    }
}
