use crate::commands::{cache, keygen, mock, prove, verify};
use afs_stark_backend::config::Com;
use afs_stark_backend::config::PcsProof;
use afs_stark_backend::config::PcsProverData;
use afs_test_utils::engine::StarkEngine;
use afs_test_utils::page_config::PageConfig;
use clap::Parser;
use clap::Subcommand;
use p3_field::PrimeField64;
use p3_uni_stark::Domain;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::de::DeserializeOwned;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(author, version, about = "AFS CLI")]
#[command(propagate_version = true)]
pub struct Cli<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[command(subcommand)]
    pub command: CliCommand<SC, E>,
}

#[derive(Debug, Subcommand)]
pub enum CliCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[command(name = "mock", about = "Mock functions")]
    /// Mock functions
    Mock(mock::MockCommand),

    #[command(name = "keygen", about = "Generate partial proving and verifying keys")]
    /// Generate partial proving and verifying keys
    Keygen(keygen::KeygenCommand<SC, E>),

    #[command(
        name = "cache",
        about = "Create the cached trace of a page from a page file"
    )]
    /// Create cached trace of a page from a page file
    Cache(cache::CacheCommand<SC, E>),

    #[command(name = "prove", about = "Generates a multi-STARK proof")]
    /// Generates a multi-STARK proof
    Prove(prove::ProveCommand<SC, E>),

    #[command(name = "verify", about = "Verifies a multi-STARK proof")]
    /// Verifies a multi-STARK proof
    Verify(verify::VerifyCommand<SC, E>),
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> Cli<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    Domain<SC>: Send + Sync,
    SC::Pcs: Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    pub fn run(config: &PageConfig, engine: &E) -> Self {
        let cli = Self::parse();
        match &cli.command {
            CliCommand::Mock(mock) => {
                mock.execute(config).unwrap();
            }
            CliCommand::Keygen(keygen) => {
                keygen.execute(config, engine).unwrap();
            }
            CliCommand::Cache(cache) => {
                cache.execute(config, engine).unwrap();
            }
            CliCommand::Prove(prove) => {
                prove.execute(config, engine).unwrap();
            }
            CliCommand::Verify(verify) => {
                verify.execute(config, engine).unwrap();
            }
        }
        cli
    }
}
