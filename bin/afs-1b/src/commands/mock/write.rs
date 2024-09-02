use std::{collections::HashSet, fs::remove_file, time::Instant};

use afs_page::page_btree::PageBTree;
use afs_stark_backend::prover::{trace::TraceCommitter, MultiTraceStarkProver};
use ax_sdk::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    page_config::MultitierPageConfig,
};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::afs_input::AfsInputFile;
use p3_util::log2_strict_usize;

use crate::commands::{
    commit_to_string, load_input_file, BABYBEAR_COMMITMENT_LEN, DECOMP_BITS, LIMB_BITS,
};

#[derive(Debug, Parser)]
pub struct WriteCommand {
    #[arg(
        long = "afi-file",
        short = 'f',
        help = "The .afi file input",
        required = true
    )]
    pub afi_file_path: String,

    #[arg(
        long = "db-folder",
        short = 'd',
        help = "Mock DB folder",
        required = false,
        default_value = "multitier_mockdb"
    )]
    pub db_folder: String,

    #[arg(
        long = "output-table-id",
        short = 'o',
        help = "Output table id (default: no output saved)",
        required = false
    )]
    pub output_table_id: Option<String>,

    #[arg(
        long = "silent",
        short = 's',
        help = "Don't print the output to stdout",
        required = false
    )]
    pub silent: bool,

    #[arg(
        long = "clean",
        short = 'c',
        help = "Delete old files if output-table-id is set",
        required = false
    )]
    pub clean: bool,
}

/// `mock write` subcommand, does unverified updates
impl WriteCommand {
    /// Execute the `mock write` command
    pub fn execute(&self, config: &MultitierPageConfig) -> Result<()> {
        let start1 = Instant::now();
        let idx_len = (config.page.index_bytes + 1) / 2;
        let data_len = (config.page.data_bytes + 1) / 2;

        let dst_id = match &self.output_table_id {
            Some(output_table_id) => output_table_id.to_owned(),
            None => "".to_owned(),
        };
        println!("afi_file_path: {}", self.afi_file_path);
        let instructions = AfsInputFile::open(&self.afi_file_path)?;
        let table_id = instructions.header.table_id.clone();
        let mut db = match PageBTree::<BABYBEAR_COMMITMENT_LEN>::load(
            self.db_folder.clone(),
            table_id.to_owned(),
            dst_id.clone(),
        ) {
            Some(t) => t,
            None => PageBTree::new(
                LIMB_BITS,
                idx_len,
                data_len,
                config.page.leaf_height,
                config.page.internal_height,
                dst_id.clone(),
            ),
        };
        load_input_file(&mut db, &instructions);
        let duration = start1.elapsed();
        println!("Wrote in memory table operations {:?}", duration);
        let start2 = Instant::now();
        if self.output_table_id.is_some() {
            let trace_degree = config.page.max_rw_ops * 4;

            let log_page_height = log2_strict_usize(config.page.leaf_height);
            let log_trace_degree = log2_strict_usize(trace_degree);

            let engine = config::baby_bear_poseidon2::default_engine(
                log_page_height.max(DECOMP_BITS).max(log_trace_degree),
            );
            let prover = MultiTraceStarkProver::new(&engine.config);
            let trace_committer = TraceCommitter::new(prover.pcs());
            if self.clean {
                let final_pages = db.gen_all_trace(&trace_committer, Some(self.db_folder.clone()));
                let init_pages = db.gen_loaded_trace();
                let mut init_leaf_set = HashSet::<Vec<u32>>::new();
                let mut init_internal_set = HashSet::<Vec<u32>>::new();
                for c in init_pages.leaf_commits {
                    init_leaf_set.insert(c.clone());
                }
                for c in init_pages.internal_commits {
                    init_internal_set.insert(c.clone());
                }
                for c in final_pages.leaf_commits {
                    if init_leaf_set.contains(&c) {
                        init_leaf_set.remove(&c);
                    }
                }
                for c in final_pages.internal_commits {
                    if init_internal_set.contains(&c) {
                        init_internal_set.remove(&c);
                    }
                }
                for c in init_leaf_set.iter() {
                    let c_str = commit_to_string(c);
                    let path = self.db_folder.clone() + "/leaf/" + &c_str;
                    remove_file(path.clone() + ".cache.bin").unwrap();
                    remove_file(path + ".trace").unwrap();
                }
                for c in init_internal_set.iter() {
                    let c_str = commit_to_string(c);
                    let path = self.db_folder.clone() + "/internal/" + &c_str;
                    remove_file(path.clone() + ".cache.bin").unwrap();
                    remove_file(path + ".trace").unwrap();
                }
            }
            db.commit::<BabyBearPoseidon2Config>(&trace_committer, self.db_folder.clone());
        }
        let duration = start2.elapsed();
        println!("Committed table operations in {:?}", duration);

        Ok(())
    }
}
