use crate::commands::run::{PageConfig, RunCommand};
use afs_chips::{common::page::Page, group_by::page_controller::PageController};
use afs_stark_backend::{
    keygen::{types::MultiStarkPartialProvingKey, MultiStarkKeygenBuilder},
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
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
    config: &PageConfig,
    cli: &RunCommand,
    db: &mut MockDb,
    op: GroupByOp,
) -> Result<()> {
    println!("group_by: {:?}", op);
    let index_bytes = config.page.index_bytes;
    let data_bytes = config.page.data_bytes;
    let idx_len = (index_bytes + 1) / 2;
    let data_len = (data_bytes + 1) / 2;
    let height = config.page.height;
    let page_width = 1 + idx_len + data_len;
    let idx_decomp = log2_strict_usize(height);

    // Get page from DB
    let mut interface = AfsInterface::new(index_bytes, data_bytes, db);
    let table = interface.get_table(op.table_id.to_string()).unwrap();
    let page = table.to_page(index_bytes, data_bytes, height);

    let mut page_controller = PageController::<BabyBearPoseidon2Config>::new(
        page_width,
        op.group_by_cols,
        op.agg_col,
        INTERNAL_BUS,
        OUTPUT_BUS,
        RANGE_BUS,
        config.page.bits_per_fe,
        idx_decomp,
    );
    let engine = config::baby_bear_poseidon2::default_engine(idx_decomp);
    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    let group_by_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let final_page_ptr =
        keygen_builder.add_cached_main_matrix(page_controller.final_chip.page_width());
    let group_by_aux_ptr = keygen_builder.add_main_matrix(page_controller.group_by.aux_width());
    let final_page_aux_ptr = keygen_builder.add_main_matrix(page_controller.final_chip.aux_width());
    let range_checker_ptr =
        keygen_builder.add_main_matrix(page_controller.range_checker.air_width());

    keygen_builder.add_partitioned_air(
        &page_controller.group_by,
        height,
        0,
        vec![group_by_ptr, group_by_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.final_chip,
        height,
        0,
        vec![final_page_ptr, final_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.range_checker.air,
        height,
        0,
        vec![range_checker_ptr],
    );

    // let pcs_log_degree = log2_ceil_usize(height);
    // let perm = random_perm();
    // let engine = engine_from_perm(perm, pcs_log_degree, config.fri_params);
    // let prover = engine.prover();
    // let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let prover = engine.prover();

    // let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    // Load a page into the GroupBy controller
    let (group_by_traces, _group_by_commitments, mut prover_data) =
        page_controller.load_page(&page, &trace_builder.committer);

    let range_checker_trace = page_controller.range_checker.generate_trace();

    trace_builder.clear();
    trace_builder.load_cached_trace(group_by_traces.group_by_trace, prover_data.remove(0));
    trace_builder.load_cached_trace(
        group_by_traces.final_page_trace.clone(),
        prover_data.remove(0),
    );
    trace_builder.load_trace(group_by_traces.group_by_aux_trace);
    trace_builder.load_trace(group_by_traces.final_page_aux_trace);
    trace_builder.load_trace(range_checker_trace);
    trace_builder.commit_current();

    let partial_pk: MultiStarkPartialProvingKey<BabyBearPoseidon2Config> =
        keygen_builder.generate_partial_pk();
    let partial_vk = partial_pk.partial_vk();

    let main_trace_data = trace_builder.view(
        &partial_vk,
        vec![
            &page_controller.group_by,
            &page_controller.final_chip,
            &page_controller.range_checker.air,
        ],
    );

    let pis = vec![vec![]; partial_vk.per_air.len()];
    let verifier = engine.verifier();

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, &partial_pk, main_trace_data, &pis);

    let mut challenger = engine.new_challenger();
    let verify = verifier.verify(
        &mut challenger,
        partial_vk,
        vec![
            &page_controller.group_by,
            &page_controller.final_chip,
            &page_controller.range_checker.air,
        ],
        proof,
        &pis,
    );

    match verify {
        Ok(_) => {
            let output_page =
                Page::from_row_major_matrix(&group_by_traces.final_page_trace, idx_len, data_len);
            if !cli.silent {
                println!("Output page: {:?}", output_page);
            }
            Ok(())
        }
        Err(e) => Err(eyre!(format!("Proof verification failed: {:?}", e))),
    }
}
