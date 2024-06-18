use super::page_controller::PageController;
use crate::common::page::Page;
use std::cmp::max;

use afs_stark_backend::{
    keygen::{types::MultiStarkPartialProvingKey, MultiStarkKeygenBuilder},
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver, USE_DEBUG_BUILDER},
    verifier::VerificationError,
};
use afs_test_utils::{
    config::{
        self,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
    },
    engine::StarkEngine,
    utils::create_seeded_rng,
};
use rand::Rng;

struct GroupByTest {
    page_width: usize,
    num_groups: usize,
    group_by_cols: Vec<usize>,
    aggregated_col: usize,

    internal_bus_index: usize,
    output_bus_index: usize,
    range_bus_index: usize,
    idx_limb_bits: usize,
    idx_decomp: usize,

    log_page_height: usize,
}

impl GroupByTest {
    pub fn new(
        page_width: usize,
        num_groups: usize,
        log_page_height: usize,
        idx_limb_bits: usize,
        idx_decomp: usize,
    ) -> Self {
        let mut my_test = GroupByTest {
            page_width,
            num_groups,
            group_by_cols: vec![],
            aggregated_col: 0,
            internal_bus_index: 0,
            output_bus_index: 1,
            range_bus_index: 2,
            idx_limb_bits,
            idx_decomp,
            log_page_height,
        };
        my_test.generate_group_by_aggregated_cols();
        my_test
    }

    fn generate_group_by_aggregated_cols(&mut self) {
        let mut rng = create_seeded_rng();
        let mut group_by_cols = vec![];

        while group_by_cols.len() < self.num_groups + 1 {
            let col = rng.gen::<usize>() % (self.page_width - 1);
            if !group_by_cols.contains(&col) {
                group_by_cols.push(col);
            }
        }
        self.aggregated_col = group_by_cols.pop().unwrap();
        self.group_by_cols = group_by_cols;
    }

    pub fn idx_len(&self) -> usize {
        self.page_width - 1
    }
    pub fn page_height(&self) -> usize {
        1 << self.log_page_height
    }
    pub fn max_idx(&self) -> usize {
        1 << self.idx_limb_bits
    }
    pub fn range_height(&self) -> usize {
        1 << self.idx_decomp
    }

    pub fn generate_page(&self, rng: &mut impl Rng, rows_allocated: usize) -> Page {
        Page::random(
            rng,
            self.idx_len(),
            0,
            self.max_idx() as u32,
            0,
            self.page_height(),
            rows_allocated,
        )
    }
    fn set_up_keygen_builder(
        &self,
        keygen_builder: &mut MultiStarkKeygenBuilder<BabyBearPoseidon2Config>,
        page_controller: &PageController<BabyBearPoseidon2Config>,
    ) {
        let group_by_ptr = keygen_builder.add_cached_main_matrix(self.page_width);
        let final_page_ptr =
            keygen_builder.add_cached_main_matrix(page_controller.final_chip.page_width());
        let group_by_aux_ptr = keygen_builder.add_main_matrix(page_controller.group_by.aux_width());
        let final_page_aux_ptr =
            keygen_builder.add_main_matrix(page_controller.final_chip.aux_width());
        let range_checker_ptr =
            keygen_builder.add_main_matrix(page_controller.range_checker.air_width());

        keygen_builder.add_partitioned_air(
            &page_controller.group_by,
            self.page_height(),
            0,
            vec![group_by_ptr, group_by_aux_ptr],
        );

        keygen_builder.add_partitioned_air(
            &page_controller.final_chip,
            self.page_height(),
            0,
            vec![final_page_ptr, final_page_aux_ptr],
        );

        keygen_builder.add_partitioned_air(
            &page_controller.range_checker.air,
            self.range_height(),
            0,
            vec![range_checker_ptr],
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn load_page_test(
    engine: &BabyBearPoseidon2Engine,
    page_init: &Page,
    page_controller: &mut PageController<BabyBearPoseidon2Config>,
    trace_builder: &mut TraceCommitmentBuilder<BabyBearPoseidon2Config>,
    partial_pk: &MultiStarkPartialProvingKey<BabyBearPoseidon2Config>,
) -> Result<(), VerificationError> {
    let (group_by_traces, _group_by_commitments, mut prover_data) =
        page_controller.load_page(page_init, &trace_builder.committer);

    let range_checker_trace = page_controller.range_checker.generate_trace();

    trace_builder.clear();

    trace_builder.load_cached_trace(group_by_traces.group_by_trace, prover_data.remove(0));
    trace_builder.load_cached_trace(group_by_traces.final_page_trace, prover_data.remove(0));
    trace_builder.load_trace(group_by_traces.group_by_aux_trace);
    trace_builder.load_trace(group_by_traces.final_page_aux_trace);
    trace_builder.load_trace(range_checker_trace);

    trace_builder.commit_current();

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

    let prover = engine.prover();
    let verifier = engine.verifier();

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, partial_pk, main_trace_data, &pis);

    let mut challenger = engine.new_challenger();

    verifier.verify(
        &mut challenger,
        partial_vk,
        vec![
            &page_controller.group_by,
            &page_controller.final_chip,
            &page_controller.range_checker.air,
        ],
        proof,
        &pis,
    )
}

#[test]
fn group_by_test() {
    let test = GroupByTest::new(20, 3, 3, 10, 4);
    let mut rng = create_seeded_rng();

    let mut page_controller = PageController::new(
        test.page_width,
        test.group_by_cols.clone(),
        test.aggregated_col,
        test.internal_bus_index,
        test.output_bus_index,
        test.range_bus_index,
        test.idx_limb_bits,
        test.idx_decomp,
    );

    let engine =
        config::baby_bear_poseidon2::default_engine(max(test.log_page_height, test.idx_decomp));
    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    test.set_up_keygen_builder(&mut keygen_builder, &page_controller);

    let partial_pk = keygen_builder.generate_partial_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let alloc_rows_arr = 0..test.page_height() - 1;

    for rows_allocated in alloc_rows_arr {
        let page = test.generate_page(&mut rng, rows_allocated);

        load_page_test(
            &engine,
            &page,
            &mut page_controller,
            &mut trace_builder,
            &partial_pk,
        )
        .expect("Verification failed");

        page_controller.refresh_range_checker();
    }
}
