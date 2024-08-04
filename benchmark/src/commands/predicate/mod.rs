use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    prover::metrics::TraceMetrics,
};
use afs_test_utils::{
    config::{
        baby_bear_blake3::BabyBearBlake3Engine,
        baby_bear_bytehash::engine_from_byte_hash,
        baby_bear_keccak::BabyBearKeccakEngine,
        baby_bear_poseidon2::{self, BabyBearPoseidon2Engine},
        baby_bear_sha256_compress::{self, BabyBearSha256CompressionEngine},
        goldilocks_poseidon::{self, GoldilocksPoseidonEngine},
        m31_sha256_compress::{self, Mersenne31Sha256CompressionEngine},
        EngineType,
    },
    engine::StarkEngine,
    page_config::PageConfig,
};
use clap::Parser;
use color_eyre::eyre::Result;
use olap::{
    commands::{
        cache::filter::CacheFilterCommand, keygen::filter::KeygenFilterCommand, parse_afo_file,
        prove::filter::ProveFilterCommand, verify::filter::VerifyFilterCommand,
    },
    KEYS_FOLDER,
};
use p3_blake3::Blake3;
use p3_field::PrimeField64;
use p3_keccak::Keccak256Hash;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};
use tracing::info_span;

use super::CommonCommands;
use crate::{DB_FILE_PATH, FILTER_FILE_PATH, TMP_FOLDER};

#[derive(Debug, Parser)]
pub struct PredicateCommand {
    #[arg(
        long = "afo-file",
        short = 'f',
        help = "Path to the .afo file",
        required = true
    )]
    pub afo_file: String,

    #[command(flatten)]
    pub common: CommonCommands,
}

impl PredicateCommand {
    pub fn bench_all<SC: StarkGenericConfig, E: StarkEngine<SC>>(
        config: &PageConfig,
        engine: &E,
        _extra_data: String,
    ) -> Result<TraceMetrics>
    where
        Val<SC>: PrimeField64,
        PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
        PcsProof<SC>: Send + Sync,
        Domain<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Pcs: Sync,
        SC::Challenge: Send + Sync,
    {
        let afo = parse_afo_file(FILTER_FILE_PATH.to_string());
        let op = afo.operations[0].clone();
        let common = olap::commands::CommonCommands {
            db_path: DB_FILE_PATH.to_string(),
            afo_path: FILTER_FILE_PATH.to_string(),
            output_path: Some(TMP_FOLDER.to_string()),
            silent: true,
        };

        // Run keygen
        let keygen_span = info_span!("Benchmark keygen").entered();
        KeygenFilterCommand::execute(config, engine, &common, op.clone(), KEYS_FOLDER.to_string())?;
        keygen_span.exit();

        // Cache span for compatibility
        let cache_span = info_span!("Benchmark cache").entered();
        CacheFilterCommand::execute(config, engine, &common, op.clone(), TMP_FOLDER.to_string())?;
        cache_span.exit();

        // Run prove
        let prove_span = info_span!("Benchmark prove").entered();
        let metrics = ProveFilterCommand::execute(
            config,
            engine,
            &common,
            op.clone(),
            KEYS_FOLDER.to_string(),
            TMP_FOLDER.to_string(),
        )?;
        prove_span.exit();

        // Run verify
        let verify_span = info_span!("Benchmark verify").entered();
        VerifyFilterCommand::execute(
            config,
            engine,
            &common,
            op.clone(),
            KEYS_FOLDER.to_string(),
            Some(TMP_FOLDER.to_string()),
            None,
        )?;
        verify_span.exit();

        Ok(metrics)
    }
}

pub fn run_bench_predicate(config: &PageConfig, extra_data: String) -> Result<TraceMetrics> {
    let fri_params = config.fri_params;
    let engine_type = config.stark_engine.engine;
    match engine_type {
        EngineType::BabyBearBlake3 => {
            let engine: BabyBearBlake3Engine = engine_from_byte_hash(Blake3, fri_params);
            PredicateCommand::bench_all(config, &engine, extra_data)
        }
        EngineType::BabyBearKeccak => {
            let engine: BabyBearKeccakEngine = engine_from_byte_hash(Keccak256Hash, fri_params);
            PredicateCommand::bench_all(config, &engine, extra_data)
        }
        EngineType::BabyBearSha256Compress => {
            let engine: BabyBearSha256CompressionEngine =
                baby_bear_sha256_compress::engine_from_fri_params(fri_params);
            PredicateCommand::bench_all(config, &engine, extra_data)
        }
        EngineType::Mersenne31Sha256Compress => {
            let engine: Mersenne31Sha256CompressionEngine =
                m31_sha256_compress::engine_from_fri_params(fri_params);
            PredicateCommand::bench_all(config, &engine, extra_data)
        }
        EngineType::BabyBearPoseidon2 => {
            let perm = baby_bear_poseidon2::default_perm();
            let engine: BabyBearPoseidon2Engine =
                baby_bear_poseidon2::engine_from_perm(perm, fri_params);
            PredicateCommand::bench_all(config, &engine, extra_data)
        }
        EngineType::GoldilocksPoseidon => {
            let perm = goldilocks_poseidon::random_perm();
            let engine: GoldilocksPoseidonEngine =
                goldilocks_poseidon::engine_from_perm(perm, fri_params);
            PredicateCommand::bench_all(config, &engine, extra_data)
        }
    }
}
