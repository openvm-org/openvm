use super::page_controller::PageController;
use crate::common::page::Page;
use crate::group_by::group_by_input::GroupByOperation;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_util::log2_strict_usize;
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

/// Struct for generating a group-by test case.
///
/// It has a random group-by column, and a random final page column.
/// It also has a random number of groups.
/// The test case is generated by generating a random page, and then loading it
/// into the group-by controller.
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
    sorted: bool,
}

impl GroupByTest {
    /// Create a new group-by test case. This is the only way to create a new
    /// group-by test case.
    pub fn new(
        page_width: usize,
        num_groups: usize,
        log_page_height: usize,
        idx_limb_bits: usize,
        idx_decomp: usize,
        sorted: bool,
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
            sorted,
        };
        my_test.generate_group_by_aggregated_cols();
        my_test
    }

    /// Randomly generate the group-by columns and the aggregated column.
    fn generate_group_by_aggregated_cols(&mut self) {
        let mut rng = create_seeded_rng();
        let mut group_by_cols = vec![];

        while group_by_cols.len() < self.num_groups + 1 {
            let col = rng.gen::<usize>() % (self.page_width - 1);
            if !group_by_cols.contains(&col) {
                group_by_cols.push(col);
            }
        }
        if self.sorted {
            group_by_cols = (0..self.num_groups + 1).collect();
        }
        self.aggregated_col = group_by_cols.pop().unwrap();
        self.group_by_cols = group_by_cols;
    }

    /// The length of indices, i.e. `page_width - 1`.
    pub fn idx_len(&self) -> usize {
        self.page_width - 1
    }

    /// The height of the page.
    pub fn page_height(&self) -> usize {
        1 << self.log_page_height
    }

    /// The maximum index value.
    pub fn max_idx(&self) -> usize {
        1 << self.idx_limb_bits
    }

    /// The height of the range checker.
    pub fn range_height(&self) -> usize {
        1 << self.idx_decomp
    }

    /// Generate a random page.
    pub fn generate_page(&self, rng: &mut impl Rng, rows_allocated: usize) -> Page {
        let mut matrix = vec![];

        // Generate half the height of unique rows
        for _ in 0..(self.page_height() / 2) {
            let mut row = vec![1];
            row.extend((0..self.idx_len()).map(|_| rng.gen::<u32>() % (1 << self.idx_limb_bits)));
            matrix.push(row);
        }

        // Add random identical copies of rows
        for _ in 0..(self.page_height() / 2) {
            let row = matrix[rng.gen_range(0..matrix.len())].clone();
            matrix.push(row);
        }

        // Adjust the 0s and 1s to match rows_allocated
        matrix.iter_mut().enumerate().for_each(|(i, row)| {
            if i >= rows_allocated {
                row[0] = 0;
            }
        });

        // Sort the matrix
        matrix.sort_by(|a, b| b.cmp(a));
        matrix[..rows_allocated].sort();
        Page::from_2d_vec(&matrix, self.idx_len(), 0)
    }

    pub fn generate_sorted_page(&self, rng: &mut impl Rng, rows_allocated: usize) -> Page {
        let page = Page::random(
            rng,
            self.idx_len(),
            0,
            self.max_idx() as u32,
            0,
            self.page_height(),
            rows_allocated,
        );
        let mut page_vecs: Vec<Vec<u32>> = page.to_2d_vec();
        page_vecs.sort_by(|a, b| b.cmp(a));
        page_vecs[..rows_allocated].sort();
        Page::from_2d_vec(&page_vecs, self.idx_len(), 0)
    }

    /// Set up the keygen builder for the group-by test case by querying trace widths.
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

    /// Load a page into the group-by controller, load traces into the `trace_builder`,
    /// generate a proof for the group-by operation, and verify it.
    ///
    /// If `perturb` is true, the page is perturbed by replacing a random value with a random value.
    #[allow(clippy::too_many_arguments)]
    fn load_page_test(
        &self,
        engine: &BabyBearPoseidon2Engine,
        page_init: &Page,
        page_controller: &mut PageController<BabyBearPoseidon2Config>,
        trace_builder: &mut TraceCommitmentBuilder<BabyBearPoseidon2Config>,
        partial_pk: &MultiStarkPartialProvingKey<BabyBearPoseidon2Config>,
        perturb: bool,
        rng: &mut impl Rng,
    ) -> Result<(), VerificationError> {
        let (group_by_traces, _group_by_commitments, mut prover_data) =
            page_controller.load_page(page_init, &trace_builder.committer);

        let range_checker_trace = page_controller.range_checker.generate_trace();

        trace_builder.clear();

        trace_builder.load_cached_trace(group_by_traces.group_by_trace, prover_data.remove(0));
        if !perturb {
            trace_builder
                .load_cached_trace(group_by_traces.final_page_trace, prover_data.remove(0));
        } else {
            trace_builder.load_cached_trace(
                perturb_page(
                    group_by_traces.final_page_trace,
                    page_init.width(),
                    rng,
                    self.max_idx(),
                ),
                prover_data.remove(0),
            );
        }
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
}

/// Perturb a page trace by randomly selecting any value in the page and replacing it with a random value.
fn perturb_page(
    trace: p3_matrix::dense::DenseMatrix<p3_baby_bear::BabyBear>,
    page_width: usize,
    rng: &mut impl Rng,
    max_value: usize,
) -> p3_matrix::dense::DenseMatrix<p3_baby_bear::BabyBear> {
    let height = trace.values.len() / trace.width;
    let perturbed_x = rng.gen_range(0..height);
    let perturbed_y = rng.gen_range(0..page_width);
    let mut perturbed_trace = trace.clone();
    perturbed_trace.values[perturbed_x * page_width + perturbed_y] =
        BabyBear::from_canonical_u32(rng.gen_range(0..max_value) as u32);
    perturbed_trace
}

#[test]
fn test_static_values() {
    let page_vec = vec![
        vec![1, 0, 0, 1, 1, 0, 0, 0, 1],
        vec![1, 0, 0, 1, 5, 0, 0, 0, 9],
        vec![1, 0, 0, 3, 9, 0, 0, 1, 5],
        vec![1, 0, 0, 3, 1, 0, 0, 9, 1],
        vec![1, 0, 1, 5, 6, 0, 0, 0, 2],
        vec![1, 1, 1, 5, 8, 0, 1, 0, 1],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    let page = Page::from_2d_vec(&page_vec, 4, 4);
    let page_width = page_vec[0].len();
    let height = page_vec.len();
    let limb_bits = 10;
    let degree = log2_strict_usize(height);
    let idx_decomp = 4;
    let internal_bus = 0;
    let output_bus = 1;
    let range_bus = 2;
    let sorted = false;
    let op = GroupByOperation::Sum;
    let mut page_controller = PageController::new(
        page_width,
        vec![2],
        3,
        internal_bus,
        output_bus,
        range_bus,
        limb_bits,
        idx_decomp,
        sorted,
        op,
    );
    let engine = config::baby_bear_poseidon2::default_engine(degree);
    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    page_controller.set_up_keygen_builder(&mut keygen_builder, height, 1 << idx_decomp);

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
    assert!(verify.is_ok());
}

#[test]
fn test_random_values() {
    let mut rng = create_seeded_rng();
    let page_width = rng.gen_range(2..20);
    let random_value = rng.gen_range(1..page_width - 1);
    let log_page_height = rng.gen_range(3..6);
    let sorted = false;
    let op = GroupByOperation::Sum;
    let test = GroupByTest::new(page_width, random_value, log_page_height, 10, 4, sorted);

    let mut page_controller = PageController::new(
        test.page_width,
        test.group_by_cols.clone(),
        test.aggregated_col,
        test.internal_bus_index,
        test.output_bus_index,
        test.range_bus_index,
        test.idx_limb_bits,
        test.idx_decomp,
        sorted,
        op,
    );

    let engine =
        config::baby_bear_poseidon2::default_engine(max(test.log_page_height, test.idx_decomp));
    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    test.set_up_keygen_builder(&mut keygen_builder, &page_controller);

    let partial_pk = keygen_builder.generate_partial_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let alloc_rows_arr: Vec<usize> = (0..test.page_height() - 1).collect();

    // Positive test
    for rows_allocated in alloc_rows_arr.iter() {
        let page = test.generate_page(&mut rng, *rows_allocated);

        test.load_page_test(
            &engine,
            &page,
            &mut page_controller,
            &mut trace_builder,
            &partial_pk,
            false,
            &mut rng,
        )
        .expect("Verification failed");

        page_controller.refresh_range_checker();
    }

    // Negative test
    for rows_allocated in alloc_rows_arr.iter() {
        let page = test.generate_page(&mut rng, *rows_allocated);

        USE_DEBUG_BUILDER.with(|debug| {
            *debug.lock().unwrap() = false;
        });

        assert_eq!(
            test.load_page_test(
                &engine,
                &page,
                &mut page_controller,
                &mut trace_builder,
                &partial_pk,
                true,
                &mut rng,
            ),
            Err(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        page_controller.refresh_range_checker();
    }
}

#[test]
fn group_by_sorted_test() {
    let mut rng = create_seeded_rng();
    let page_width = rng.gen_range(2..20);
    let num_groups = page_width - 2;
    let log_page_height = rng.gen_range(3..6);
    let sorted = true;
    let op = GroupByOperation::Sum;
    let test = GroupByTest::new(page_width, num_groups, log_page_height, 10, 4, sorted);

    let mut page_controller = PageController::new(
        test.page_width,
        test.group_by_cols.clone(),
        test.aggregated_col,
        test.internal_bus_index,
        test.output_bus_index,
        test.range_bus_index,
        test.idx_limb_bits,
        test.idx_decomp,
        sorted,
        op,
    );

    let engine =
        config::baby_bear_poseidon2::default_engine(max(test.log_page_height, test.idx_decomp));
    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    page_controller.set_up_keygen_builder(
        &mut keygen_builder,
        test.page_height(),
        1 << test.idx_decomp,
    );

    let partial_pk = keygen_builder.generate_partial_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let alloc_rows_arr: Vec<usize> = (0..test.page_height() - 1).collect();

    // Positive test
    for rows_allocated in alloc_rows_arr.iter() {
        let page = test.generate_sorted_page(&mut rng, *rows_allocated);

        test.load_page_test(
            &engine,
            &page,
            &mut page_controller,
            &mut trace_builder,
            &partial_pk,
            false,
            &mut rng,
        )
        .expect("Verification failed");

        page_controller.refresh_range_checker();
    }

    // Negative test
    for rows_allocated in alloc_rows_arr.iter() {
        let page = test.generate_sorted_page(&mut rng, *rows_allocated);

        USE_DEBUG_BUILDER.with(|debug| {
            *debug.lock().unwrap() = false;
        });

        assert_eq!(
            test.load_page_test(
                &engine,
                &page,
                &mut page_controller,
                &mut trace_builder,
                &partial_pk,
                true,
                &mut rng,
            ),
            Err(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        page_controller.refresh_range_checker();
    }
}
