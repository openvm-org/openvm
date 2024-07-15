use std::marker::PhantomData;

use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;
use p3_field::PrimeField64;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};

use self::{
    keygen::InnerJoinKeygenCommand, prove::InnerJoinProveCommand, verify::InnerJoinVerifyCommand,
};

pub mod keygen;
pub mod prove;
pub mod verify;

#[derive(Debug, Parser)]
pub struct InnerJoinCommonCommands {
    #[arg(
        long = "db-path",
        short = 'd',
        help = "The path to the database",
        required = true
    )]
    pub db_path: String,

    #[arg(
        long = "afo-path",
        short = 'f',
        help = "The path to the .afo file",
        required = true
    )]
    pub afo_path: String,

    #[arg(
        long = "output-path",
        short = 'o',
        help = "The path to the output file",
        required = false
    )]
    pub output_path: Option<String>,

    #[arg(
        long = "silent",
        short = 's',
        help = "Don't print the output to stdout",
        required = false
    )]
    pub silent: bool,
}

#[derive(Debug, Parser)]
pub struct InnerJoinCli<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[command(subcommand)]
    pub command: InnerJoinCommand<SC, E>,

    #[command(flatten)]
    pub common: InnerJoinCommonCommands,

    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

#[derive(Debug, Subcommand)]
pub enum InnerJoinCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[command(name = "keygen", about = "Generate keys for inner join")]
    Keygen(InnerJoinKeygenCommand<SC, E>),

    #[command(name = "prove", about = "Prove inner join")]
    Prove(prove::InnerJoinProveCommand<SC, E>),

    #[command(name = "verify", about = "Verify inner join")]
    Verify(verify::InnerJoinVerifyCommand<SC, E>),
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> InnerJoinCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Sync,
    SC::Challenge: Send + Sync,
{
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        common: &InnerJoinCommonCommands,
        command: &InnerJoinCommand<SC, E>,
    ) -> Result<()> {
        match command {
            InnerJoinCommand::Keygen(cmd) => {
                InnerJoinKeygenCommand::execute(config, engine, common).unwrap();
            }
            InnerJoinCommand::Prove(cmd) => {
                InnerJoinProveCommand::execute(config, engine, common).unwrap();
            }
            InnerJoinCommand::Verify(cmd) => {
                InnerJoinVerifyCommand::execute(config, engine, common).unwrap();
            }
        }
        Ok(())
    }
}
