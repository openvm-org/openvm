use afs::commands::{
    cache::CacheCommand, keygen::KeygenCommand, prove::ProveCommand, verify::VerifyCommand,
};
use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use afs_test_utils::{
    config::{
        baby_bear_blake3::BabyBearBlake3Engine,
        baby_bear_bytehash::engine_from_byte_hash,
        baby_bear_keccak::BabyBearKeccakEngine,
        baby_bear_poseidon2::{self, BabyBearPoseidon2Engine},
        goldilocks_poseidon::{self, GoldilocksPoseidonEngine},
        EngineType,
    },
    engine::StarkEngine,
    page_config::PageConfig,
};
use clap::Parser;
use color_eyre::eyre::Result;
use p3_blake3::Blake3;
use p3_field::PrimeField64;
use p3_keccak::Keccak256Hash;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
use serde::{de::DeserializeOwned, Serialize};
use tracing::info_span;

use crate::{AFI_FILE_PATH, DB_FILE_PATH, TABLE_ID, TMP_FOLDER};

use super::CommonCommands;

#[derive(Debug, Parser)]
pub struct RwCommand {
    #[arg(
        long = "percent-reads",
        short = 'r',
        help = "Percentage of max_rw_ops that are reads (100 = 100%)",
        default_value = "50",
        required = true
    )]
    /// Percentage of max_rw_ops that are reads (100 = 100%)
    pub percent_reads: usize,

    #[arg(
        long = "percent-writes",
        short = 'w',
        help = "Percentage of max_rw_ops that are writes (100 = 100%)",
        default_value = "50",
        required = true
    )]
    /// Percentage of max_rw_ops that are writes (100 = 100%)
    pub percent_writes: usize,

    #[command(flatten)]
    pub common: CommonCommands,
}

impl RwCommand {
    pub fn bench_all<SC: StarkGenericConfig, E: StarkEngine<SC>>(
        config: &PageConfig,
        engine: &E,
        _extra_data: String,
    ) -> Result<()>
    where
        Val<SC>: PrimeField64,
        PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
        PcsProof<SC>: Send + Sync,
        Domain<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Pcs: Sync,
        SC::Challenge: Send + Sync,
    {
        let proof_file = DB_FILE_PATH.to_string() + ".prove.bin";

        // Run keygen
        let keygen_span = info_span!("Benchmark keygen").entered();
        KeygenCommand::execute(config, engine, TMP_FOLDER.to_string())?;
        keygen_span.exit();

        // Run cache
        let cache_span = info_span!("Benchmark cache").entered();
        CacheCommand::execute(
            config,
            engine,
            TABLE_ID.to_string(),
            DB_FILE_PATH.to_string(),
            TMP_FOLDER.to_string(),
        )?;
        cache_span.exit();

        // Run prove
        let prove_span = info_span!("Benchmark prove").entered();
        ProveCommand::execute(
            config,
            engine,
            AFI_FILE_PATH.to_string(),
            DB_FILE_PATH.to_string(),
            TMP_FOLDER.to_string(),
            TMP_FOLDER.to_string(),
            true,
        )?;
        prove_span.exit();

        // Run verify
        let verify_span = info_span!("Benchmark verify").entered();
        VerifyCommand::execute(
            config,
            engine,
            proof_file,
            DB_FILE_PATH.to_string(),
            TMP_FOLDER.to_string(),
        )?;
        verify_span.exit();

        Ok(())
    }
}

pub fn run_bench_rw(config: &PageConfig, extra_data: String) -> Result<()> {
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
            RwCommand::bench_all(config, &engine, extra_data)
        }
        EngineType::BabyBearKeccak => {
            let engine: BabyBearKeccakEngine =
                engine_from_byte_hash(Keccak256Hash, pcs_log_degree, fri_params);
            RwCommand::bench_all(config, &engine, extra_data)
        }
        EngineType::BabyBearPoseidon2 => {
            let perm = baby_bear_poseidon2::default_perm();
            let engine: BabyBearPoseidon2Engine =
                baby_bear_poseidon2::engine_from_perm(perm, pcs_log_degree, fri_params);
            RwCommand::bench_all(config, &engine, extra_data)
        }
        EngineType::GoldilocksPoseidon => {
            let perm = goldilocks_poseidon::random_perm();
            let engine: GoldilocksPoseidonEngine =
                goldilocks_poseidon::engine_from_perm(perm, pcs_log_degree, fri_params);
            RwCommand::bench_all(config, &engine, extra_data)
        }
    }
}
