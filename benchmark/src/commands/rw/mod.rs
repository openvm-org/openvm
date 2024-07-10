use std::{collections::HashMap, fs, time::Instant};

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
use chrono::Local;
use clap::Parser;
use color_eyre::eyre::Result;
use p3_blake3::Blake3;
use p3_field::PrimeField64;
use p3_keccak::Keccak256Hash;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
use serde::{de::DeserializeOwned, Serialize};
use tracing::info_span;

use crate::{
    commands::parse_config_folder,
    utils::{
        config_gen::generate_configs,
        output_writer::{save_afi_to_new_db, write_csv_header, write_csv_line},
        random_table::generate_random_afi_rw,
        tracing::{clear_tracing_log, extract_event_data_from_log, extract_timing_data_from_log},
    },
    AFI_FILE_PATH, DB_FILE_PATH, DEFAULT_OUTPUT_FILE, TABLE_ID, TMP_FOLDER, TMP_TRACING_LOG,
};

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
    pub fn execute(&self) -> Result<()> {
        println!("Executing Read/Write benchmark");

        assert!(self.percent_reads + self.percent_writes <= 100);

        // Generate/Parse config(s)
        let configs = if let Some(config_folder) = self.common.config_folder.clone() {
            parse_config_folder(config_folder)
        } else {
            generate_configs()
        };
        let configs_len = configs.len();

        // Create tmp folder
        let _ = fs::create_dir_all(TMP_FOLDER);

        // Write .csv file
        let output_file = self
            .common
            .output_file
            .clone()
            .unwrap_or(DEFAULT_OUTPUT_FILE.to_string());
        write_csv_header(output_file.clone())?;
        println!("Output file: {}", output_file.clone());

        // Parse engine
        for (idx, config) in configs.iter().rev().enumerate() {
            let timestamp = Local::now().format("%H:%M:%S");
            println!(
                "[{}] Running config {:?}: {} of {}",
                timestamp,
                config.generate_filename(),
                idx + 1,
                configs_len
            );

            clear_tracing_log(TMP_TRACING_LOG.as_str())?;

            // Generate AFI file
            let generate_afi_instant = Instant::now();
            generate_random_afi_rw(
                config,
                TABLE_ID.to_string(),
                AFI_FILE_PATH.to_string(),
                self.percent_reads,
                self.percent_writes,
            )?;
            let generate_afi_duration = generate_afi_instant.elapsed();
            println!("Setup: generate AFI duration: {:?}", generate_afi_duration);

            // Save AFI file data to database
            let save_afi_instant = Instant::now();
            save_afi_to_new_db(config, AFI_FILE_PATH.to_string(), DB_FILE_PATH.to_string())?;
            let save_afi_duration = save_afi_instant.elapsed();
            println!("Setup: save AFI to DB duration: {:?}", save_afi_duration);

            run_rw_bench(config).unwrap();

            let event_data = extract_event_data_from_log(
                TMP_TRACING_LOG.as_str(),
                &[
                    "Total air width: preprocessed=",
                    "Total air width: partitioned_main=",
                    "Total air width: after_challenge=",
                ],
            )?;
            let timing_data = extract_timing_data_from_log(
                TMP_TRACING_LOG.as_str(),
                &[
                    "ReadWrite keygen",
                    "ReadWrite cache",
                    "ReadWrite prove",
                    "prove:Load page trace generation: afs_chips::page_rw_checker::page_controller",
                    "prove:Load page trace commitment: afs_chips::page_rw_checker::page_controller",
                    "Prove.generate_trace",
                    "prove:Prove trace commitment",
                    "ReadWrite verify",
                ],
            )?;

            println!("Config: {:?}", config);
            println!("Event data: {:?}", event_data);
            println!("Timing data: {:?}", timing_data);
            println!("Output file: {}", output_file.clone());

            let mut log_data: HashMap<String, String> = event_data;
            log_data.extend(timing_data);

            write_csv_line(
                output_file.clone(),
                "ReadWrite".to_string(),
                config,
                &log_data,
                self.percent_reads,
                self.percent_writes,
            )?;
        }

        println!("Benchmark ReadWrite completed.");

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
        let proof_file = DB_FILE_PATH.to_string() + ".prove.bin";

        // Run keygen
        let keygen_span = info_span!("ReadWrite keygen").entered();
        KeygenCommand::execute(config, engine, TMP_FOLDER.to_string())?;
        keygen_span.exit();

        // Run cache
        let cache_span = info_span!("ReadWrite cache").entered();
        CacheCommand::execute(
            config,
            engine,
            TABLE_ID.to_string(),
            DB_FILE_PATH.to_string(),
            TMP_FOLDER.to_string(),
        )?;
        cache_span.exit();

        // Run prove
        let prove_span = info_span!("ReadWrite prove").entered();
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
        let verify_span = info_span!("ReadWrite verify").entered();
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
