use std::{str::FromStr, time::Instant};

use afs_chips::single_page_index_scan::{
    page_controller::PageController, page_index_scan_input::Comp,
};
use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
};
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
    utils::{fixed_bytes_to_field_vec, string_to_fixed_bytes_be_vec},
};
use p3_util::log2_strict_usize;

use super::common::CommonCommands;

const PAGE_BUS_INDEX: usize = 0;
const RANGE_BUS_INDEX: usize = 1;

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
        let start = Instant::now();
        let idx_len = config.page.index_bytes / 2;
        let data_len = config.page.data_bytes / 2;
        let page_height = config.page.height;
        let idx_limb_bits = config.page.bits_per_fe;
        let idx_decomp = log2_strict_usize(page_height).max(8);

        let mut db = MockDb::from_file(self.args.db_file_path.unwrap().as_str());
        let mut interface = AfsInterface::<U256, U256>::new(&mut db);
        let table_id = self.args.table_id;
        let table = interface.get_table(table_id.clone()).unwrap();
        let page_input = table.to_page(page_height);
        let value = self.args.value;
        let cmp = Comp::Eq;

        // Handle the page and predicate in ZK and get the resulting page back
        let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
            PAGE_BUS_INDEX,
            RANGE_BUS_INDEX,
            idx_len,
            data_len,
            page_height as u32,
            idx_limb_bits,
            idx_decomp,
            cmp.clone(),
        );

        let page_width = 1 + idx_len + data_len;
        let value = string_to_fixed_bytes_be_vec(value, config.page.index_bytes);
        let value = fixed_bytes_to_field_vec(value);
        let page_output =
            page_controller.gen_output(page_input.clone(), value.clone(), page_width, cmp);

        let engine = config::baby_bear_poseidon2::default_engine(idx_decomp);
        let prover = MultiTraceStarkProver::new(&engine.config);
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        page_controller.load_page(
            page_input.clone(),
            page_output.clone(),
            value.clone(),
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
            &mut trace_builder.committer,
        );

        let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);
        page_controller.set_up_keygen_builder(
            &mut keygen_builder,
            page_width,
            page_height,
            idx_len,
            idx_decomp,
        );

        let partial_pk = keygen_builder.generate_partial_pk();
        let partial_vk = partial_pk.partial_vk();
        let proof = page_controller.prove(
            &engine,
            &partial_pk,
            &mut trace_builder,
            value.clone(),
            idx_decomp,
        );
        page_controller
            .verify(&engine, partial_vk, proof, value.clone())
            .unwrap();

        // Convert back to a Table
        let table_id_bytes = TableId::from_str(table_id.as_str())?;
        let table_output = Table::<U256, U256>::from_page(table_id_bytes, page_output);

        let duration = start.elapsed();
        // Save the output to a file or print it
        if !self.args.silent {
            println!("Table ID: {}", table_id);
            println!("{:?}", table.metadata);
            for (index, data) in table_output.body.iter() {
                println!("{:?}: {:?}", index, data);
            }
            println!("Proved predicate operations in {:?}", duration);
        }

        Ok(())
    }
}
