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
            rows_allocated,
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

    // // We expect failure, so we turn off debug assertions
    // USE_DEBUG_BUILDER.with(|debug| {
    //     *debug.lock().unwrap() = false;
    // });
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
    let test = GroupByTest::new(4, 1, 2, 10, 4);
    // let test = GroupByTest::new(20, 3, 3, 10, 4);
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

    let page = test.generate_page(&mut rng, test.page_height());

    // Testing a fully allocated page
    load_page_test(
        &engine,
        &page,
        &mut page_controller,
        &mut trace_builder,
        &partial_pk,
    )
    .expect("Verification failed");

    let page = test.generate_page(&mut rng, test.page_height() - 1);

    // Testing a half allocated page
    load_page_test(
        &engine,
        &page,
        &mut page_controller,
        &mut trace_builder,
        &partial_pk,
    )
    .expect("Verification failed");

    // let rows_allocated = rng.gen::<usize>() % (test.page_height() + 1);
    // for i in rows_allocated..test.page_height() {
    //     page[i][0] = 0;

    //     // Making sure the first operation using this index is a write
    //     let idx = page[i][1..test.idx_len() + 1].to_vec();
    //     for op in ops.iter_mut() {
    //         if op.idx == idx {
    //             op.op_type = OpType::Write;
    //             break;
    //         }
    //     }
    // }

    // load_page_test(
    //     &engine,
    //     &page,
    //     &mut page_controller,
    //     &mut trace_builder,
    //     &partial_pk,
    // )
    // .expect("Verification failed");

    // // Testing a fully unallocated page
    // for i in 0..page_height {
    //     // Making sure the first operation that uses every index is a write
    //     let idx = page[i][1..idx_len + 1].to_vec();
    //     for op in ops.iter_mut() {
    //         if op.idx == idx {
    //             op.op_type = OpType::Write;
    //             break;
    //         }
    //     }

    //     let idx: Vec<u32> = (0..idx_len).map(|_| rng.gen::<u32>() % max_idx).collect();
    //     let data: Vec<u32> = (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect();
    //     page[i] = iter::once(0).chain(idx).chain(data).collect();
    // }

    // load_page_test(
    //     &engine,
    //     page.clone(),
    //     idx_len,
    //     data_len,
    //     idx_limb_bits,
    //     idx_decomp,
    //     &ops,
    //     &mut page_controller,
    //     &ops_sender,
    //     &mut trace_builder,
    //     &partial_pk,
    //     trace_degree,
    //     num_ops,
    // )
    // .expect("Verification failed");

    // // Testing writing only 1 index into an unallocated page
    // ops = vec![Operation::new(
    //     10,
    //     (0..idx_len).map(|_| rng.gen::<u32>() % max_idx).collect(),
    //     (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect(),
    //     OpType::Write,
    // )];

    // load_page_test(
    //     &engine,
    //     page.clone(),
    //     idx_len,
    //     data_len,
    //     idx_limb_bits,
    //     idx_decomp,
    //     &ops,
    //     &mut page_controller,
    //     &ops_sender,
    //     &mut trace_builder,
    //     &partial_pk,
    //     trace_degree,
    //     num_ops,
    // )
    // .expect("Verification failed");

    // // Negative tests

    // // Testing reading from a non-existing index (in a fully-unallocated page)
    // ops = vec![Operation::new(
    //     1,
    //     (0..idx_len).map(|_| rng.gen::<u32>() % max_idx).collect(),
    //     (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect(),
    //     OpType::Read,
    // )];

    // USE_DEBUG_BUILDER.with(|debug| {
    //     *debug.lock().unwrap() = false;
    // });
    // assert_eq!(
    //     load_page_test(
    //         &engine,
    //         page.clone(),
    //         idx_len,
    //         data_len,
    //         idx_limb_bits,
    //         idx_decomp,
    //         &ops,
    //         &mut page_controller,
    //         &ops_sender,
    //         &mut trace_builder,
    //         &partial_pk,
    //         trace_degree,
    //         num_ops,
    //     ),
    //     Err(VerificationError::OodEvaluationMismatch),
    //     "Expected constraints to fail"
    // );

    // // Testing reading wrong data from an existing index
    // let idx: Vec<u32> = (0..idx_len).map(|_| rng.gen::<u32>() % max_idx).collect();
    // let data_1: Vec<u32> = (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect();
    // let mut data_2 = data_1.clone();
    // data_2[0] += 1; // making sure data_2 is different

    // ops = vec![
    //     Operation::new(1, idx.clone(), data_1, OpType::Write),
    //     Operation::new(2, idx, data_2, OpType::Read),
    // ];

    // assert_eq!(
    //     load_page_test(
    //         &engine,
    //         page.clone(),
    //         idx_len,
    //         data_len,
    //         idx_limb_bits,
    //         idx_decomp,
    //         &ops,
    //         &mut page_controller,
    //         &ops_sender,
    //         &mut trace_builder,
    //         &partial_pk,
    //         trace_degree,
    //         num_ops,
    //     ),
    //     Err(VerificationError::OodEvaluationMismatch),
    //     "Expected constraints to fail"
    // );

    // // Testing writing too many indices to a fully unallocated page
    // let mut idx_map = HashSet::new();
    // for _ in 0..page_height + 1 {
    //     let mut idx: Vec<u32>;
    //     loop {
    //         idx = (0..idx_len).map(|_| rng.gen::<u32>() % max_idx).collect();
    //         if !idx_map.contains(&idx) {
    //             break;
    //         }
    //     }

    //     idx_map.insert(idx);
    // }

    // ops.clear();
    // for (i, idx) in idx_map.iter().enumerate() {
    //     ops.push(Operation::new(
    //         i + 1,
    //         idx.clone(),
    //         (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect(),
    //         OpType::Write,
    //     ));
    // }

    // let engine_ref = &engine;
    // let result = panic::catch_unwind(move || {
    //     let _ = load_page_test(
    //         engine_ref,
    //         page.clone(),
    //         idx_len,
    //         data_len,
    //         idx_limb_bits,
    //         idx_decomp,
    //         &ops,
    //         &mut page_controller,
    //         &ops_sender,
    //         &mut trace_builder,
    //         &partial_pk,
    //         trace_degree,
    //         num_ops,
    //     );
    // });

    // assert!(
    //     result.is_err(),
    //     "Expected to fail when allocating too many indices"
    // );
}
