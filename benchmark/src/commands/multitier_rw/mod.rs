use std::{fs::remove_dir_all, path::Path};

use afs_1b::commands::{
    keygen::KeygenCommand, prove::ProveCommand, verify::VerifyCommand, BABYBEAR_COMMITMENT_LEN,
    DECOMP_BITS,
};
use afs_page::page_btree::PageBTree;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
};
use afs_test_utils::{
    config::{
        baby_bear_poseidon2::{engine_from_perm, random_perm, BabyBearPoseidon2Engine},
        EngineType,
    },
    engine::StarkEngine,
    page_config::MultitierPageConfig,
};
use clap::Parser;
use color_eyre::eyre::Result;
use p3_field::{PrimeField, PrimeField32, PrimeField64};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
use serde::{de::DeserializeOwned, Serialize};
use tracing::info_span;

use crate::{AFI_FILE_PATH, DB_FOLDER, KEY_FOLDER, MULTITIER_TABLE_ID};

use super::CommonCommands;

#[derive(Debug, Parser)]
pub struct MultitierRwCommand {
    #[arg(
        long = "start-config",
        short = 'i',
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
    /// Whether we do the benchmark on a new tree or an existing one.
    pub new_tree: bool,
    #[command(flatten)]
    pub common: CommonCommands,
}

impl MultitierRwCommand {
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
        let keygen_span = info_span!("Benchmark keygen").entered();
        KeygenCommand::execute(config, engine, KEY_FOLDER.to_string())?;
        keygen_span.exit();

        // Run prove
        let prove_span = info_span!("Benchmark prove").entered();
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
        let verify_span = info_span!("Benchmark verify").entered();
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
        _ => panic!(),
    }
}
