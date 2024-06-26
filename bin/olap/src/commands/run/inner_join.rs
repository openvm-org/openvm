use crate::commands::run::{utils::pretty_print_page, PageConfig, RunCommand};
use afs_chips::{
    common::page::Page,
    inner_join::controller::{FKInnerJoinController, IJBuses, TableFormat},
};
use afs_stark_backend::{keygen::MultiStarkKeygenBuilder, prover::trace::TraceCommitmentBuilder};
use afs_test_utils::{config::baby_bear_poseidon2, engine::StarkEngine};
use color_eyre::eyre::{eyre, Result};
use logical_interface::{
    afs_input::operation::InnerJoinOp, afs_interface::AfsInterface, mock_db::MockDb,
};
use p3_uni_stark::StarkGenericConfig;
use p3_util::log2_strict_usize;

const RANGE_BUS: usize = 0;
const T1_INTERSECTOR_BUS: usize = 1;
const T2_INTERSECTOR_BUS: usize = 2;
const INTERSECTOR_T2_BUS: usize = 3;
const T1_OUTPUT_BUS: usize = 4;
const T2_OUTPUT_BUS: usize = 5;

pub fn execute_inner_join<SC: StarkGenericConfig>(
    cfg: &PageConfig,
    cli: &RunCommand,
    db: &mut MockDb,
    op: InnerJoinOp,
) -> Result<()> {
    println!("inner_join: {:?}", op);

    let height = cfg.page.height;
    let limb_bits = cfg.page.bits_per_fe;
    let degree = log2_strict_usize(height);
    let range_chip_idx_decomp = 4;

    // Get input pages from database
    let interface_left = AfsInterface::new_with_table(op.table_id_left.to_string(), db);
    let table_left = interface_left.current_table().unwrap();
    let page_left = table_left.to_page(
        table_left.metadata.index_bytes,
        table_left.metadata.data_bytes,
        height,
    );
    let index_len_left = (table_left.metadata.index_bytes + 1) / 2;
    let data_len_left = (table_left.metadata.data_bytes + 1) / 2;

    let interface_right = AfsInterface::new_with_table(op.table_id_right.to_string(), db);
    let table_right = interface_right.current_table().unwrap();
    let page_right = table_right.to_page(
        table_right.metadata.index_bytes,
        table_right.metadata.data_bytes,
        height,
    );
    let index_len_right = (table_right.metadata.index_bytes + 1) / 2;
    let data_len_right = (table_right.metadata.data_bytes + 1) / 2;

    if !cli.silent {
        println!("Left page:");
        pretty_print_page(&page_left);
        println!("Right page:");
        pretty_print_page(&page_right);
    }

    let inner_join_buses = IJBuses {
        range_bus_index: RANGE_BUS,
        t1_intersector_bus_index: T1_INTERSECTOR_BUS,
        t2_intersector_bus_index: T2_INTERSECTOR_BUS,
        intersector_t2_bus_index: INTERSECTOR_T2_BUS,
        t1_output_bus_index: T1_OUTPUT_BUS,
        t2_output_bus_index: T2_OUTPUT_BUS,
    };
    let t1_format = TableFormat::new(index_len_left, data_len_left, limb_bits);
    let t2_format = TableFormat::new(index_len_right, data_len_right, limb_bits);

    let mut inner_join_controller = FKInnerJoinController::new(
        inner_join_buses,
        t1_format,
        t2_format,
        op.fkey_start,
        op.fkey_end,
        range_chip_idx_decomp,
    );
    let engine = baby_bear_poseidon2::default_engine(degree + 1);
    let prover = engine.prover();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    let prover_trace_data = inner_join_controller.load_tables(
        &page_left,
        &page_right,
        2 * height,
        &mut trace_builder.committer,
    );

    let output_trace = &inner_join_controller.traces().output_main_trace;
    let output_page = Page::from_trace(
        output_trace,
        index_len_right,
        data_len_right + data_len_left,
    );
    if !cli.silent {
        println!("Proof verified");
        println!("Output page:");
        pretty_print_page(&output_page);
    }

    let mut keygen_builder = MultiStarkKeygenBuilder::new(engine.config());
    inner_join_controller.set_up_keygen_builder(&mut keygen_builder, height, height, 2 * height);
    let partial_pk = keygen_builder.generate_partial_pk();
    let partial_vk = partial_pk.partial_vk();

    let proof =
        inner_join_controller.prove(&engine, &partial_pk, &mut trace_builder, prover_trace_data);

    let verify = inner_join_controller.verify(&engine, partial_vk, proof);

    match verify {
        Ok(_) => Ok(()),
        Err(e) => Err(eyre!(format!("Proof verification failed: {:?}", e))),
    }
}
