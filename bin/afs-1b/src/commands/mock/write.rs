use afs_chips::pagebtree::PageBTree;
use afs_stark_backend::prover::{trace::TraceCommitter, MultiTraceStarkProver};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    page_config::MultitierPageConfig,
};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::afs_input_instructions::AfsInputInstructions;
use p3_util::log2_strict_usize;

use crate::commands::{
    load_input_file, BABYBEAR_COMMITMENT_LEN, DECOMP_BITS, INTERNAL_HEIGHT, LEAF_HEIGHT, LIMB_BITS,
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
}

/// `mock read` subcommand
impl WriteCommand {
    /// Execute the `mock read` command
    pub fn execute(&self, config: &MultitierPageConfig) -> Result<()> {
        let idx_len = (config.page.index_bytes + 1) / 2 as usize;
        let data_len = (config.page.data_bytes + 1) / 2 as usize;

        let page_height = config.page.height;
        let dst_id = match &self.output_table_id {
            Some(output_table_id) => output_table_id.to_owned(),
            None => "".to_owned(),
        };
        println!("afi_file_path: {}", self.afi_file_path);
        let instructions = AfsInputInstructions::from_file(&self.afi_file_path)?;
        let table_id = instructions.header.table_id.clone();
        let mut db = match PageBTree::<INTERNAL_HEIGHT, LEAF_HEIGHT, BABYBEAR_COMMITMENT_LEN>::load(
            self.db_folder.clone(),
            table_id.to_owned(),
            dst_id.clone(),
        ) {
            Some(t) => t,
            None => PageBTree::new(
                LIMB_BITS,
                idx_len,
                data_len,
                page_height,
                page_height,
                dst_id.clone(),
            ),
        };
        load_input_file(&mut db, &instructions);

        if self.output_table_id.is_some() {
            let page_height = config.page.height;

            let trace_degree = config.page.max_rw_ops * 4;

            let log_page_height = log2_strict_usize(page_height);
            let log_trace_degree = log2_strict_usize(trace_degree);

            let engine = config::baby_bear_poseidon2::default_engine(
                log_page_height.max(DECOMP_BITS).max(log_trace_degree),
            );
            let prover = MultiTraceStarkProver::new(&engine.config);
            let mut trace_committer = TraceCommitter::new(prover.pcs());
            db.commit::<BabyBearPoseidon2Config>(&mut trace_committer, self.db_folder.clone());
        }

        Ok(())
    }
}
