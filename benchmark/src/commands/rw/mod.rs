use std::fs;

use afs::commands::{
    cache::CacheCommand, keygen::KeygenCommand, prove::ProveCommand, verify::VerifyCommand,
};
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
use p3_blake3::Blake3;
use p3_field::PrimeField64;
use p3_keccak::Keccak256Hash;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    commands::parse_configs, random_table::generate_random_table, DB_FILE_PATH, TABLE_ID,
    TMP_FOLDER,
};

use super::CommonCommands;

#[derive(Debug, Parser)]
pub struct RwCommand {
    #[command(flatten)]
    pub common: CommonCommands,
}

impl RwCommand {
    pub fn execute(&self) -> Result<()> {
        println!("Executing Read/Write benchmark");

        // Parse config(s)
        let configs = parse_configs(self.common.config_files.clone());

        // Create tmp folder
        let _ = fs::create_dir_all(TMP_FOLDER);

        // Parse engine
        for config in configs {
            // Generate and save random table to db
            generate_random_table(&config, TABLE_ID.to_string(), DB_FILE_PATH.to_string());
            run_rw_bench(&config).unwrap();
        }

        // Write .csv file
        let _output_file = self.common.output_file.clone();

        Ok(())
    }

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
        let afi_file_path = TMP_FOLDER.to_string() + "/instructions.afi";
        let proof_file = DB_FILE_PATH.to_string() + ".prove.bin";

        // Run keygen
        KeygenCommand::execute(config, engine, TMP_FOLDER.to_string())?;

        // Run cache
        CacheCommand::execute(
            config,
            engine,
            TABLE_ID.to_string(),
            DB_FILE_PATH.to_string(),
            TMP_FOLDER.to_string(),
        )?;

        // Run prove
        ProveCommand::execute(
            config,
            engine,
            afi_file_path,
            DB_FILE_PATH.to_string(),
            TMP_FOLDER.to_string(),
            TMP_FOLDER.to_string(),
            false,
        )?;

        // Run verify
        VerifyCommand::execute(
            config,
            engine,
            proof_file,
            DB_FILE_PATH.to_string(),
            TMP_FOLDER.to_string(),
        )?;

        Ok(())
    }
}

pub fn run_rw_bench(config: &PageConfig) -> Result<()> {
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
            RwCommand::bench_all(config, &engine)
        }
        EngineType::BabyBearKeccak => {
            let engine: BabyBearKeccakEngine =
                engine_from_byte_hash(Keccak256Hash, pcs_log_degree, fri_params);
            RwCommand::bench_all(config, &engine)
        }
        EngineType::BabyBearPoseidon2 => {
            let perm = random_perm();
            let engine: BabyBearPoseidon2Engine =
                engine_from_perm(perm, pcs_log_degree, fri_params);
            RwCommand::bench_all(config, &engine)
        }
    }
}
