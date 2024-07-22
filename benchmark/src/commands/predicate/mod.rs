use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use afs_test_utils::{
    config::{
        baby_bear_blake3::BabyBearBlake3Engine,
        baby_bear_bytehash::engine_from_byte_hash,
        baby_bear_keccak::BabyBearKeccakEngine,
        baby_bear_poseidon2::{engine_from_perm, random_perm, BabyBearPoseidon2Engine},
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
use p3_util::log2_strict_usize;
use serde::{de::DeserializeOwned, Serialize};
use tracing::info_span;

use crate::{DB_FILE_PATH, FILTER_FILE_PATH, TMP_FOLDER};

use super::CommonCommands;

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
        KeygenFilterCommand::execute(config, engine, &common, op.clone(), KEYS_FOLDER.to_string())
            .unwrap();
        keygen_span.exit();

        // Cache span for compatibility
        let cache_span = info_span!("Benchmark cache").entered();
        CacheFilterCommand::execute(config, engine, &common, op.clone(), TMP_FOLDER.to_string())
            .unwrap();
        cache_span.exit();

        // Run prove
        let prove_span = info_span!("Benchmark prove").entered();
        ProveFilterCommand::execute(
            config,
            engine,
            &common,
            op.clone(),
            KEYS_FOLDER.to_string(),
            TMP_FOLDER.to_string(),
        )
        .unwrap();
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
        )
        .unwrap();
        verify_span.exit();

        Ok(())
    }
}

pub fn run_predicate_bench(config: &PageConfig, extra_data: String) -> Result<()> {
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
            PredicateCommand::bench_all(config, &engine, extra_data)
        }
        EngineType::BabyBearKeccak => {
            let engine: BabyBearKeccakEngine =
                engine_from_byte_hash(Keccak256Hash, pcs_log_degree, fri_params);
            PredicateCommand::bench_all(config, &engine, extra_data)
        }
        EngineType::BabyBearPoseidon2 => {
            let perm = random_perm();
            let engine: BabyBearPoseidon2Engine =
                engine_from_perm(perm, pcs_log_degree, fri_params);
            PredicateCommand::bench_all(config, &engine, extra_data)
        }
    }
}
