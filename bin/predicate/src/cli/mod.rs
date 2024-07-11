use afs_stark_backend::config::Com;
use afs_stark_backend::config::PcsProof;
use afs_stark_backend::config::PcsProverData;
use afs_stark_backend::config::StarkGenericConfig;
use afs_stark_backend::config::Val;
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
use p3_util::log2_strict_usize;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::commands::{
    keygen::KeygenCommand, prove::ProveCommand, verify::VerifyCommand, CommonCommands,
};

#[derive(Debug, Parser)]
#[command(author, version, about = "AFS Predicate CLI")]
#[command(propagate_version = true)]
pub struct Cli<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[command(subcommand)]
    pub command: CliCommand<SC, E>,
}

#[derive(Debug, Subcommand)]
pub enum CliCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[command(name = "keygen", about = "Generate keys")]
    Keygen(KeygenCommand<SC, E>),

    #[command(name = "prove", about = "Prove the predicate operation")]
    Prove(ProveCommand<SC, E>),

    #[command(name = "verify", about = "Verify the predicate operation")]
    Verify(VerifyCommand<SC, E>),
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
            CliCommand::Keygen(keygen) => {
                let common = CommonCommands {
                    predicate: keygen.common.predicate.clone(),
                    cache_folder: keygen.common.cache_folder.clone(),
                    output_folder: keygen.common.output_folder.clone(),
                    silent: keygen.common.silent,
                };
                KeygenCommand::execute(config, engine, &common).unwrap();
            }
            CliCommand::Prove(prove) => {
                let common = CommonCommands {
                    predicate: prove.common.predicate.clone(),
                    cache_folder: prove.common.cache_folder.clone(),
                    output_folder: prove.common.output_folder.clone(),
                    silent: prove.common.silent,
                };
                ProveCommand::execute(
                    config,
                    engine,
                    &common,
                    prove.value.clone(),
                    prove.table_id.clone(),
                    prove.db_file_path.clone(),
                    prove.keys_folder.clone(),
                    prove.input_trace_file.clone(),
                    prove.output_trace_folder.clone(),
                )
                .unwrap();
            }
            CliCommand::Verify(verify) => {
                let common = CommonCommands {
                    predicate: verify.common.predicate.clone(),
                    cache_folder: verify.common.cache_folder.clone(),
                    output_folder: verify.common.output_folder.clone(),
                    silent: verify.common.silent,
                };
                VerifyCommand::execute(
                    config,
                    engine,
                    &common,
                    verify.value.clone(),
                    verify.table_id.clone(),
                    verify.keys_folder.clone(),
                )
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
