use std::{
    collections::HashMap,
    fs::{self, remove_dir_all},
    path::Path,
    time::Instant,
};

use afs_1b::commands::{
    keygen::KeygenCommand, prove::ProveCommand, verify::VerifyCommand, BABYBEAR_COMMITMENT_LEN,
    DECOMP_BITS,
};
use afs_chips::page_btree::PageBTree;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
};
use afs_test_utils::{
    config::{
        baby_bear_blake3::BabyBearBlake3Engine,
        baby_bear_bytehash::engine_from_byte_hash,
        baby_bear_keccak::BabyBearKeccakEngine,
        baby_bear_poseidon2::{engine_from_perm, random_perm, BabyBearPoseidon2Engine},
        EngineType,
    },
    engine::StarkEngine,
    page_config::MultitierPageConfig,
};
use chrono::Local;
use clap::Parser;
use color_eyre::eyre::Result;
use p3_blake3::Blake3;
use p3_field::{PrimeField, PrimeField32, PrimeField64};
use p3_keccak::Keccak256Hash;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
use serde::{de::DeserializeOwned, Serialize};
use tracing::info_span;

use crate::{
    commands::{parse_config_folder, parse_multitier_config_folder},
    utils::{
        config_gen::{generate_configs, generate_multitier_configs},
        output_writer::{
            save_afi_to_new_db, write_csv_header, write_csv_line, write_multitier_csv_header,
            write_multitier_csv_line,
        },
        table_gen::{generate_random_afi_rw, generate_random_multitier_afi_rw},
        tracing::{clear_tracing_log, extract_event_data_from_log, extract_timing_data_from_log},
    },
    AFI_FILE_PATH, DB_FILE_PATH, DB_FOLDER, KEY_FOLDER, MULTITIER_TABLE_ID, TABLE_ID, TMP_FOLDER,
    TMP_TRACING_LOG,
};

use super::CommonCommands;

#[derive(Debug, Parser)]
pub struct MultitierRwCommand {
    // #[arg(
    //     long = "percent-reads",
    //     short = 'r',
    //     help = "Percentage of max_rw_ops that are reads (100 = 100%)",
    //     default_value = "50",
    //     required = true
    // )]
    // /// Percentage of max_rw_ops that are reads (100 = 100%)
    // pub percent_reads: usize,
    #[arg(
        long = "start-config",
        short = 's',
        help = "Choose to start a certain config",
        default_value = "0",
        required = true
    )]
    pub start_idx: usize,

    #[arg(
        long = "new-tree",
        short = 'n',
        help = "Choose to start with a new tree or a large tree",
        required = true
    )]
    /// Percentage of max_rw_ops that are writes (100 = 100%)
    pub new_tree: bool,
    #[command(flatten)]
    pub common: CommonCommands,
}

impl MultitierRwCommand {
    // pub fn execute(&self) -> Result<()> {
    //     println!("Executing Multitier Read/Write benchmark");

    //     // assert!(self.percent_reads + self.percent_writes <= 100);

    //     // Generate/Parse config(s)
    //     let configs = if let Some(config_folder) = self.common.config_folder.clone() {
    //         parse_multitier_config_folder(config_folder)
    //     } else {
    //         generate_multitier_configs()
    //     };
    //     let configs_len = configs.len();

    //     // Create tmp folder
    //     let _ = fs::create_dir_all(TMP_FOLDER);

    //     // Write .csv file
    //     let output_file = self
    //         .common
    //         .output_file
    //         .clone()
    //         .unwrap_or(self.common.output_file);
    //     write_multitier_csv_header(output_file.clone())?;
    //     println!("Output file: {}", output_file.clone());

    //     // Parse engine
    //     for (idx, config) in configs.iter().rev().enumerate() {
    //         let timestamp = Local::now().format("%H:%M:%S");
    //         println!(
    //             "[{}] Running config {:?}: {} of {}",
    //             timestamp,
    //             config.generate_filename(),
    //             idx + 1,
    //             configs_len
    //         );

    //         clear_tracing_log(TMP_TRACING_LOG.as_str())?;

    //         // Generate AFI file
    //         let generate_afi_instant = Instant::now();
    //         generate_random_multitier_afi_rw(
    //             config,
    //             MULTITIER_TABLE_ID.to_string(),
    //             AFI_FILE_PATH.to_string(),
    //         )?;
    //         let generate_afi_duration = generate_afi_instant.elapsed();
    //         println!("Setup: generate AFI duration: {:?}", generate_afi_duration);

    //         run_rw_bench(config, self.new_tree).unwrap();

    //         let event_data = extract_event_data_from_log(
    //             TMP_TRACING_LOG.as_str(),
    //             &[
    //                 "Total air width: preprocessed=",
    //                 "Total air width: partitioned_main=",
    //                 "Total air width: after_challenge=",
    //             ],
    //         )?;
    //         let timing_data = extract_timing_data_from_log(
    //             TMP_TRACING_LOG.as_str(),
    //             &[
    //                 "ReadWrite keygen",
    //                 "ReadWrite prove",
    //                 "Page BTree Updates",
    //                 "Page BTree Commit to Disk",
    //                 "Page BTree Load Traces and Prover Data",
    //                 "prove:Load page trace generation: afs_chips::multitier_page_rw_checker::page_controller",
    //                 "Prove.generate_trace",
    //                 "prove:Prove trace commitment",
    //                 "ReadWrite verify",
    //             ],
    //         )?;

    //         println!("Config: {:?}", config);
    //         println!("Event data: {:?}", event_data);
    //         println!("Timing data: {:?}", timing_data);
    //         println!("Output file: {}", output_file.clone());

    //         let mut log_data: HashMap<String, String> = event_data;
    //         log_data.extend(timing_data);

    //         write_multitier_csv_line(
    //             output_file.clone(),
    //             "Multitier ReadWrite".to_string(),
    //             scenario,
    //             config,
    //             &log_data,
    //         )?;
    //     }

    //     println!("Benchmark ReadWrite completed.");

    //     Ok(())
    // }

    pub fn bench_all<SC: StarkGenericConfig, E: StarkEngine<SC>>(
        config: &MultitierPageConfig,
        engine: &E,
        new_tree: bool,
    ) -> Result<()>
    where
        Val<SC>: PrimeField + PrimeField64 + PrimeField32,
        Com<SC>: Into<[Val<SC>; BABYBEAR_COMMITMENT_LEN]>,
        PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
        PcsProof<SC>: Send + Sync,
        Domain<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Pcs: Sync,
        SC::Challenge: Send + Sync,
    {
        let idx_len = (config.page.index_bytes + 1) / 2;
        let data_len = (config.page.data_bytes + 1) / 2;
        if new_tree {
            let prover = MultiTraceStarkProver::new(engine.config());
            let trace_builder = TraceCommitmentBuilder::<SC>::new(prover.pcs());
            let db_folder_path = DB_FOLDER.clone();
            let db_folder_path = Path::new(&db_folder_path);
            if db_folder_path.is_dir() {
                remove_dir_all(DB_FOLDER.to_string()).unwrap();
            }
            let key_folder_path = KEY_FOLDER.clone();
            let key_folder_path = Path::new(&key_folder_path);
            if key_folder_path.is_dir() {
                remove_dir_all(KEY_FOLDER.to_string()).unwrap();
            }
            let mut init_tree = PageBTree::<BABYBEAR_COMMITMENT_LEN>::new(
                config.page.bits_per_fe,
                idx_len,
                data_len,
                config.page.leaf_height,
                config.page.internal_height,
                MULTITIER_TABLE_ID.to_string(),
            );
            init_tree.commit(&trace_builder.committer, DB_FOLDER.to_string());
        }
        // Run keygen
        let keygen_span = info_span!("ReadWrite keygen").entered();
        KeygenCommand::execute(config, engine, KEY_FOLDER.to_string())?;
        keygen_span.exit();

        // Run prove
        let prove_span = info_span!("ReadWrite prove").entered();
        ProveCommand::execute(
            config,
            engine,
            AFI_FILE_PATH.to_string(),
            DB_FOLDER.to_string(),
            KEY_FOLDER.to_string(),
            true,
        )?;
        prove_span.exit();

        // Run verify
        let verify_span = info_span!("ReadWrite verify").entered();
        VerifyCommand::execute(
            config,
            engine,
            MULTITIER_TABLE_ID.to_string(),
            DB_FOLDER.to_string(),
            KEY_FOLDER.to_string(),
        )?;
        verify_span.exit();

        Ok(())
    }
}

pub fn run_mtrw_bench(config: &MultitierPageConfig, new_tree: String) -> Result<()> {
    let new_tree = new_tree == "true";
    let checker_trace_degree = config.page.max_rw_ops * 4;
    let pcs_log_degree = log2_strict_usize(checker_trace_degree)
        .max(log2_strict_usize(config.page.leaf_height))
        .max(DECOMP_BITS);
    let fri_params = config.fri_params;
    let engine_type = config.stark_engine.engine;
    match engine_type {
        EngineType::BabyBearBlake3 => {
            panic!()
        }
        EngineType::BabyBearKeccak => {
            panic!()
        }
        EngineType::BabyBearPoseidon2 => {
            let perm = random_perm();
            let engine: BabyBearPoseidon2Engine =
                engine_from_perm(perm, pcs_log_degree, fri_params);
            MultitierRwCommand::bench_all(config, &engine, new_tree)
        }
    }
}
