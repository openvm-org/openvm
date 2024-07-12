use afs::commands::cache::CacheCommand;
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
use logical_interface::afs_interface::utils::string_to_table_id;
use p3_blake3::Blake3;
use p3_field::PrimeField64;
use p3_keccak::Keccak256Hash;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
use predicate::commands::{keygen::KeygenCommand, prove::ProveCommand, verify::VerifyCommand};
use serde::{de::DeserializeOwned, Serialize};
use tracing::info_span;

use crate::{DB_FILE_PATH, TABLE_ID, TMP_FOLDER};

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
        extra_data: String,
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
        let (predicate, value) = {
            let split: Vec<&str> = extra_data.split_whitespace().collect();
            (split[2].to_string(), split[3].to_string())
        };
        let common = predicate::commands::CommonCommands {
            predicate,
            cache_folder: TMP_FOLDER.to_string(),
            output_folder: TMP_FOLDER.to_string(),
            silent: true,
        };
        let input_trace_file = format!(
            "{}/{}.cache.bin",
            TMP_FOLDER,
            string_to_table_id(TABLE_ID.to_string())
        );

        // Run keygen
        let keygen_span = info_span!("Benchmark keygen").entered();
        KeygenCommand::execute(config, engine, &common).unwrap();
        keygen_span.exit();

        // Cache span for compatibility
        let cache_span = info_span!("Benchmark cache").entered();
        CacheCommand::execute(
            config,
            engine,
            TABLE_ID.to_string(),
            DB_FILE_PATH.to_string(),
            TMP_FOLDER.to_string(),
        )
        .unwrap();
        cache_span.exit();

        // Run prove
        let prove_span = info_span!("Benchmark prove").entered();
        ProveCommand::execute(
            config,
            engine,
            &common,
            value.clone(),
            TABLE_ID.to_string(),
            DB_FILE_PATH.to_string(),
            TMP_FOLDER.to_string(),
            input_trace_file,
            TMP_FOLDER.to_string(),
        )
        .unwrap();
        prove_span.exit();

        // Run verify
        let verify_span = info_span!("Benchmark verify").entered();
        VerifyCommand::execute(
            config,
            engine,
            &common,
            value.clone(),
            TABLE_ID.to_string(),
            TMP_FOLDER.to_string(),
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
