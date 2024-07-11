use std::{collections::HashMap, time::Instant};

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
use chrono::Local;
use clap::Parser;
use color_eyre::eyre::Result;
use p3_blake3::Blake3;
use p3_field::PrimeField64;
use p3_keccak::Keccak256Hash;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    commands::benchmark_setup,
    utils::{
        output_writer::{save_afi_to_new_db, write_csv_line},
        table_gen::generate_random_afi_rw,
        tracing::{clear_tracing_log, extract_event_data_from_log, extract_timing_data_from_log},
    },
    AFI_FILE_PATH, DB_FILE_PATH, TABLE_ID, TMP_TRACING_LOG,
};

use super::CommonCommands;

#[derive(Debug, Parser)]
pub struct PredicateCommand {
    #[arg(
        long = "predicate",
        short = 'p',
        help = "Predicate to run",
        required = true
    )]
    pub predicate: String,

    #[arg(
        long = "value",
        short = 'v',
        help = "Value to prove the predicate against",
        required = true
    )]
    pub value: String,

    #[command(flatten)]
    pub common: CommonCommands,
}

impl PredicateCommand {
    pub fn bench_all<SC: StarkGenericConfig, E: StarkEngine<SC>>(
        config: &PageConfig,
        engine: &E,
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
        Ok(())
    }
}

pub fn run_predicate_bench(config: &PageConfig) -> Result<()> {
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
            PredicateCommand::bench_all(config, &engine)
        }
        EngineType::BabyBearKeccak => {
            let engine: BabyBearKeccakEngine =
                engine_from_byte_hash(Keccak256Hash, pcs_log_degree, fri_params);
            PredicateCommand::bench_all(config, &engine)
        }
        EngineType::BabyBearPoseidon2 => {
            let perm = random_perm();
            let engine: BabyBearPoseidon2Engine =
                engine_from_perm(perm, pcs_log_degree, fri_params);
            PredicateCommand::bench_all(config, &engine)
        }
    }
}
