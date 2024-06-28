use crate::commands::run::{utils::pretty_print_page, PageConfig, RunCommand};
use afs_chips::{common::page::Page, group_by::page_controller::PageController};
use afs_stark_backend::{
    keygen::{types::MultiStarkPartialProvingKey, MultiStarkKeygenBuilder},
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
};
use afs_test_utils::{
    config::baby_bear_poseidon2::{self, BabyBearPoseidon2Config},
    engine::StarkEngine,
};
use color_eyre::eyre::{eyre, Result};
use logical_interface::{
    afs_input::operation::GroupByOp, afs_interface::AfsInterface, mock_db::MockDb,
};
use p3_uni_stark::StarkGenericConfig;
use p3_util::log2_strict_usize;

const INTERNAL_BUS: usize = 0;
const OUTPUT_BUS: usize = 1;
const RANGE_BUS: usize = 2;

pub fn execute_group_by<SC: StarkGenericConfig>(
    cfg: &PageConfig,
    cli: &RunCommand,
    db: &mut MockDb,
    op: GroupByOp,
) -> Result<()> {
    println!("group_by: {:?}", op);
    let index_bytes = cfg.page.index_bytes;
    let data_bytes = cfg.page.data_bytes;
    let height = 16; //cfg.page.height;
    let limb_bits = 10; //cfg.page.bits_per_fe;
    let idx_len = (index_bytes + 1) / 2;
    let data_len = (data_bytes + 1) / 2;
    let page_width = 1 + idx_len + data_len;
    let degree = log2_strict_usize(height);
    let decomp = 4;

    println!("group_by cols: {:?}", op.group_by_cols);
    println!("agg col: {:?}", op.agg_col);

    // Get page from DB
    let interface = AfsInterface::new_with_table(op.table_id.to_string(), db);
    let table = interface.current_table().unwrap();
    let page = table.to_page(
        table.metadata.index_bytes,
        table.metadata.data_bytes,
        height,
    );

    if !cli.silent {
        println!("Input page");
        pretty_print_page(&page);
    }

    let mut page_controller = PageController::<BabyBearPoseidon2Config>::new(
        page_width,
        op.group_by_cols.clone(),
        op.agg_col,
        INTERNAL_BUS,
        OUTPUT_BUS,
        RANGE_BUS,
        limb_bits,
        decomp,
    );
    page_controller.refresh_range_checker();
    let engine = baby_bear_poseidon2::default_engine(degree + 1);

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);
    page_controller.set_up_keygen_builder(&mut keygen_builder, height, 1 << decomp);

    let prover = engine.prover();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    let (group_by_traces, _group_by_commitments, prover_data) =
        page_controller.load_page(&page, &trace_builder.committer);

    let partial_pk = keygen_builder.generate_partial_pk();
    let partial_vk = partial_pk.partial_vk();
    let proof = page_controller.prove(
        &engine,
        &partial_pk,
        &mut trace_builder,
        group_by_traces,
        prover_data,
    );

    let verify = page_controller.verify(&engine, partial_vk, proof);

    // Keygen
    // let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);
    // let group_by_ptr = keygen_builder.add_cached_main_matrix(page_width);
    // let final_page_ptr =
    //     keygen_builder.add_cached_main_matrix(page_controller.final_chip.page_width());
    // let group_by_aux_ptr = keygen_builder.add_main_matrix(page_controller.group_by.aux_width());
    // let final_page_aux_ptr = keygen_builder.add_main_matrix(page_controller.final_chip.aux_width());
    // let range_checker_ptr =
    //     keygen_builder.add_main_matrix(page_controller.range_checker.air_width());

    // keygen_builder.add_partitioned_air(
    //     &page_controller.group_by,
    //     height,
    //     0,
    //     vec![group_by_ptr, group_by_aux_ptr],
    // );

    // keygen_builder.add_partitioned_air(
    //     &page_controller.final_chip,
    //     height,
    //     0,
    //     vec![final_page_ptr, final_page_aux_ptr],
    // );

    // keygen_builder.add_partitioned_air(
    //     &page_controller.range_checker.air,
    //     decomp,
    //     0,
    //     vec![range_checker_ptr],
    // );

    // let prover = engine.prover();
    // let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    // // Load a page into the GroupBy controller
    // let (group_by_traces, _group_by_commitments, mut prover_data) =
    //     page_controller.load_page(&page, &trace_builder.committer);

    // let output_page =
    //     Page::from_trace(&group_by_traces.final_page_trace, op.group_by_cols.len(), 1);
    // if !cli.silent {
    //     println!("Output page");
    //     pretty_print_page(&output_page);
    // }

    // let range_checker_trace = page_controller.range_checker.generate_trace();

    // // Build trace
    // trace_builder.clear();
    // trace_builder.load_cached_trace(group_by_traces.group_by_trace, prover_data.remove(0));
    // trace_builder.load_cached_trace(
    //     group_by_traces.final_page_trace.clone(),
    //     prover_data.remove(0),
    // );
    // trace_builder.load_trace(group_by_traces.group_by_aux_trace);
    // trace_builder.load_trace(group_by_traces.final_page_aux_trace);
    // trace_builder.load_trace(range_checker_trace);
    // trace_builder.commit_current();

    // let partial_pk = keygen_builder.generate_partial_pk();
    // let partial_vk = partial_pk.partial_vk();

    // let main_trace_data = trace_builder.view(
    //     &partial_vk,
    //     vec![
    //         &page_controller.group_by,
    //         &page_controller.final_chip,
    //         &page_controller.range_checker.air,
    //     ],
    // );

    // // Prove
    // let mut challenger = engine.new_challenger();
    // let pis = vec![vec![]; partial_vk.per_air.len()];
    // let proof = prover.prove(&mut challenger, &partial_pk, main_trace_data, &pis);

    // // Verify
    // let mut challenger = engine.new_challenger();
    // let verifier = engine.verifier();
    // let verify = verifier.verify(
    //     &mut challenger,
    //     partial_vk,
    //     vec![
    //         &page_controller.group_by,
    //         &page_controller.final_chip,
    //         &page_controller.range_checker.air,
    //     ],
    //     proof,
    //     &pis,
    // );

    match verify {
        Ok(_) => Ok(()),
        Err(e) => Err(eyre!(format!("Proof verification failed: {:?}", e))),
    }
}
