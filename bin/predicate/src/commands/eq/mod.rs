use std::str::FromStr;

use afs_chips::{is_equal::IsEqualAir, page_rw_checker::page_controller::PageController};
use afs_stark_backend::prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    page_config::PageConfig,
};
use alloy_primitives::U256;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_interface::AfsInterface,
    mock_db::MockDb,
    table::{types::TableId, Table},
};
use p3_util::log2_strict_usize;

use super::common::CommonCommands;

#[derive(Debug, Parser)]
pub struct EqCommand {
    #[command(flatten)]
    pub args: CommonCommands,
}

/// `eq` command
impl EqCommand {
    /// Execute the `eq` command
    pub fn execute(self, config: &PageConfig) -> Result<()> {
        println!(
            "Running `index == {}` command on table: {}",
            self.args.value, self.args.table_id
        );
        let idx_len = (config.page.index_bytes + 1) / 2;
        let data_len = (config.page.data_bytes + 1) / 2;
        let height = config.page.height;
        let page_bus_index = 0;
        let range_bus_index = 1;
        let ops_bus_index = 2;
        let idx_limb_bits = config.page.bits_per_fe;
        let idx_decomp = 8;
        let checker_trace_degree = config.page.max_rw_ops * 4;
        let max_log_degree = log2_strict_usize(checker_trace_degree)
            .max(log2_strict_usize(height))
            .max(8);

        let mut db = MockDb::from_file(self.args.db_file_path.unwrap().as_str());
        let mut interface = AfsInterface::<U256, U256>::new(&mut db);
        let table_id = self.args.table_id;
        let table = interface.get_table(table_id.clone()).unwrap();
        let page_init = table.to_page(height);
        let predicate = "==";
        let value = self.args.value;

        // Handle the page and predicate in ZK and get the resulting page back
        let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
            page_bus_index,
            range_bus_index,
            ops_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
        );
        let ops_sender = IsEqualAir {};
        let engine = config::baby_bear_poseidon2::default_engine(max_log_degree);
        let prover = MultiTraceStarkProver::new(&engine.config);
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        // Convert back to a Table
        let table_id_bytes = TableId::from_str(table_id.as_str())?;
        let output_table = Table::<U256, U256>::from_page(table_id_bytes, page);

        // Save the output to a file or print it
        if !self.args.silent {
            println!("Table ID: {}", table_id);
            println!("{:?}", table.metadata);
            for (index, data) in table.body.iter() {
                println!("{:?}: {:?}", index, data);
            }
        }

        Ok(())
    }
}
