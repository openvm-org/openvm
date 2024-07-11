use crate::commands::cache::CacheCommand;
use crate::commands::keygen::KeygenCommand;
use crate::commands::prove::ProveCommand;
use crate::commands::verify::VerifyCommand;
use crate::commands::{cache, keygen, mock, prove, verify};
use afs_stark_backend::config::Com;
use afs_stark_backend::config::PcsProof;
use afs_stark_backend::config::PcsProverData;
use afs_test_utils::config::baby_bear_blake3::BabyBearBlake3Engine;
use afs_test_utils::config::baby_bear_bytehash::engine_from_byte_hash;
use afs_test_utils::config::baby_bear_keccak::BabyBearKeccakEngine;
use afs_test_utils::config::baby_bear_poseidon2::engine_from_perm;
use afs_test_utils::config::baby_bear_poseidon2::random_perm;
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
{
    pub fn run_with_engine(config: &PageConfig, engine: &E)
    where
        E: StarkEngine<SC>,
    {
        let cli = Self::parse();
        match &cli.command {
            CliCommand::Mock(mock) => {
                mock.execute(config).unwrap();
            }
            CliCommand::Keygen(keygen) => {
                let output_folder = keygen.output_folder.clone();
                KeygenCommand::execute(config, engine, output_folder).unwrap();
            }
            CliCommand::Cache(cache) => {
                let table_id = cache.table_id.clone();
                let db_file_path = cache.db_file_path.clone();
                let output_folder = cache.output_folder.clone();
                CacheCommand::execute(config, engine, table_id, db_file_path, output_folder)
                    .unwrap();
            }
            CliCommand::Prove(prove) => {
                let afi_file_path = prove.afi_file_path.clone();
                let db_file_path = prove.db_file_path.clone();
                let keys_folder = prove.keys_folder.clone();
                let cache_folder = prove.cache_folder.clone();
                let silent = prove.silent;
                ProveCommand::execute(
                    config,
                    engine,
                    afi_file_path,
                    db_file_path,
                    keys_folder,
                    cache_folder,
                    silent,
                )
                .unwrap();
            }
            CliCommand::Verify(verify) => {
                let proof_file = verify.proof_file.clone();
                let init_db_file_path = verify.init_db_file_path.clone();
                let keys_folder = verify.keys_folder.clone();
                VerifyCommand::execute(config, engine, proof_file, init_db_file_path, keys_folder)
                    .unwrap();
            }
        }
    }
}

pub fn run(config: &PageConfig) {
    let checker_trace_degree = config.page.max_rw_ops * 4;
    let pcs_log_degree = log2_strict_usize(checker_trace_degree)
        .max(log2_strict_usize(config.page.height))
        .max(8);
    let fri_params = config.fri_params;
    let engine_type = config.stark_engine.engine;
    match engine_type {
        EngineType::BabyBearBlake3 => {
            let engine: BabyBearBlake3Engine =
                engine_from_byte_hash(Blake3, pcs_log_degree, fri_params);
            Cli::run_with_engine(config, &engine)
        }
        EngineType::BabyBearKeccak => {
            let engine: BabyBearKeccakEngine =
                engine_from_byte_hash(Keccak256Hash, pcs_log_degree, fri_params);
            Cli::run_with_engine(config, &engine)
        }
        EngineType::BabyBearPoseidon2 => {
            let perm = random_perm();
            let engine: BabyBearPoseidon2Engine =
                engine_from_perm(perm, pcs_log_degree, fri_params);
            Cli::run_with_engine(config, &engine)
        }
    }
}
