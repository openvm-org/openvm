use crate::commands::{cache, keygen, mock, prove, verify};
use afs_stark_backend::config::Com;
use afs_stark_backend::config::PcsProof;
use afs_stark_backend::config::PcsProverData;
use afs_test_utils::config::baby_bear_blake3::BabyBearBlake3Config;
use afs_test_utils::config::baby_bear_blake3::BabyBearBlake3Engine;
use afs_test_utils::config::baby_bear_bytehash::engine_from_byte_hash;
use afs_test_utils::config::baby_bear_keccak::BabyBearKeccakConfig;
use afs_test_utils::config::baby_bear_keccak::BabyBearKeccakEngine;
use afs_test_utils::config::baby_bear_poseidon2::config_from_perm;
use afs_test_utils::config::baby_bear_poseidon2::default_engine;
use afs_test_utils::config::baby_bear_poseidon2::engine_from_perm;
use afs_test_utils::config::baby_bear_poseidon2::random_perm;
use afs_test_utils::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use afs_test_utils::config::baby_bear_poseidon2::BabyBearPoseidon2Engine;
use afs_test_utils::config::EngineType;
use afs_test_utils::engine::StarkEngine;
use afs_test_utils::page_config::PageConfig;
use clap::Parser;
use clap::Subcommand;
use p3_blake3::Blake3;
use p3_field::PrimeField64;
use p3_keccak::Keccak256Hash;
use p3_uni_stark::Domain;
use p3_uni_stark::{StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
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
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Sync,
    SC::Challenge: Send + Sync,
    BabyBearBlake3Engine: StarkEngine<BabyBearBlake3Config>,
    BabyBearKeccakEngine: StarkEngine<BabyBearKeccakConfig>,
    BabyBearPoseidon2Engine: StarkEngine<BabyBearPoseidon2Config>,
{
    pub fn run(config: &PageConfig) -> Self {
        let pcs_log_degree = log2_strict_usize(config.page.height);
        let fri_params = config.fri_params;
        let engine_type = config.stark_engine.engine;
        match engine_type {
            EngineType::BabyBearBlake3 => {
                let engine: BabyBearBlake3Engine =
                    engine_from_byte_hash(Blake3, pcs_log_degree, fri_params);
                Self::run_with_engine(config, &engine)
            }

            EngineType::BabyBearKeccak => {
                let engine: BabyBearKeccakEngine =
                    engine_from_byte_hash(Keccak256Hash, pcs_log_degree, fri_params);
                Self::run_with_engine(config, &engine)
            }
            EngineType::BabyBearPoseidon2 => {
                let perm = random_perm();
                // let config = config_from_perm(&perm, pcs_log_degree, fri_params);
                let engine: BabyBearPoseidon2Engine =
                    engine_from_perm(perm, pcs_log_degree, fri_params);
                Self::run_with_engine(config, &engine)
            }
        }
    }

    pub fn run_with_engine(config: &PageConfig, engine: &E) -> Self
    where
        E: StarkEngine<SC>,
    {
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
