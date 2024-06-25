use crate::commands::run::{PageConfig, RunCommand};
use afs_chips::group_by::page_controller::PageController;
use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    engine::StarkEngine,
};
use color_eyre::eyre::Result;
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
    let mut keygen_builder =
        MultiStarkKeygenBuilder::<BabyBearPoseidon2Config>::new(&engine.config);

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

    let partial_pk = keygen_builder.generate_partial_pk();
    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    // Load a page into the GroupBy controller
    page_controller.load_page(&page, &trace_builder.committer);

    Ok(())
}
