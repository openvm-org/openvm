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
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::{page_controller::PageController, page_index_scan_input::Comp};

#[allow(clippy::too_many_arguments)]
fn index_scan_test(
    engine: &BabyBearPoseidon2Engine,
    page: Vec<Vec<u32>>,
    page_output: Vec<Vec<u32>>,
    x: Vec<u32>,
    idx_len: usize,
    data_len: usize,
    idx_limb_bits: Vec<usize>,
    idx_decomp: usize,
    page_controller: &mut PageController<BabyBearPoseidon2Config>,
    trace_builder: &mut TraceCommitmentBuilder<BabyBearPoseidon2Config>,
    partial_pk: &MultiStarkPartialProvingKey<BabyBearPoseidon2Config>,
) -> Result<(), VerificationError> {
    let page_height = page.len();
    assert!(page_height > 0);

    let (page_traces, mut prover_data) = page_controller.load_page(
        page.clone(),
        page_output.clone(),
        x.clone(),
        idx_len,
        data_len,
        idx_limb_bits,
        idx_decomp,
        &mut trace_builder.committer,
    );

    let input_chip_aux_trace = page_controller.input_chip_aux_trace();
    let output_chip_aux_trace = page_controller.output_chip_aux_trace();
    let range_checker_trace = page_controller.range_checker_trace();

    // Clearing the range_checker counts
    page_controller.update_range_checker(idx_decomp);

    trace_builder.clear();

    trace_builder.load_cached_trace(page_traces[0].clone(), prover_data.remove(0));
    trace_builder.load_cached_trace(page_traces[1].clone(), prover_data.remove(0));
    trace_builder.load_trace(input_chip_aux_trace);
    trace_builder.load_trace(output_chip_aux_trace);
    trace_builder.load_trace(range_checker_trace);

    trace_builder.commit_current();

    let partial_vk = partial_pk.partial_vk();

    let main_trace_data = trace_builder.view(
        &partial_vk,
        vec![
            &page_controller.input_chip.air,
            &page_controller.output_chip.air,
            &page_controller.range_checker.air,
        ],
    );

    let pis = vec![
        x.iter().map(|x| BabyBear::from_canonical_u32(*x)).collect(),
        vec![],
        vec![],
    ];

    let prover = engine.prover();
    let verifier = engine.verifier();

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, partial_pk, main_trace_data, &pis);

    let mut challenger = engine.new_challenger();

    verifier.verify(
        &mut challenger,
        partial_vk,
        vec![
            &page_controller.input_chip.air,
            &page_controller.output_chip.air,
            &page_controller.range_checker.air,
        ],
        proof,
        &pis,
    )
}

#[test]
fn test_single_page_index_scan_lt() {
    let bus_index: usize = 0;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: Vec<usize> = vec![16, 16];
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;
    let page_height = 1 << log_page_height;
    let page_width = 1 + idx_len + data_len;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits.clone(),
        decomp,
        Comp::Lt,
    );

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    let input_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let output_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let input_page_aux_ptr = keygen_builder.add_main_matrix(page_controller.input_chip.aux_width());
    let output_page_aux_ptr =
        keygen_builder.add_main_matrix(page_controller.output_chip.aux_width());
    let range_checker_ptr =
        keygen_builder.add_main_matrix(page_controller.range_checker.air_width());

    keygen_builder.add_partitioned_air(
        &page_controller.input_chip.air,
        page_height,
        idx_len,
        vec![input_page_ptr, input_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.output_chip.air,
        page_height,
        0,
        vec![output_page_ptr, output_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.range_checker.air,
        1 << decomp,
        0,
        vec![range_checker_ptr],
    );

    let partial_pk = keygen_builder.generate_partial_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];

    let x: Vec<u32> = vec![2177, 5880];

    let page_output =
        page_controller.gen_output(page.clone(), x.clone(), idx_len, page_width, Comp::Lt);

    index_scan_test(
        &engine,
        page,
        page_output,
        x,
        idx_len,
        data_len,
        limb_bits,
        decomp,
        &mut page_controller,
        &mut trace_builder,
        &partial_pk,
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_lte() {
    let bus_index: usize = 0;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: Vec<usize> = vec![16, 16];
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;
    let page_height = 1 << log_page_height;
    let page_width = 1 + idx_len + data_len;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits.clone(),
        decomp,
        Comp::Lte,
    );

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    let input_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let output_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let input_page_aux_ptr = keygen_builder.add_main_matrix(page_controller.input_chip.aux_width());
    let output_page_aux_ptr =
        keygen_builder.add_main_matrix(page_controller.output_chip.aux_width());
    let range_checker_ptr =
        keygen_builder.add_main_matrix(page_controller.range_checker.air_width());

    keygen_builder.add_partitioned_air(
        &page_controller.input_chip.air,
        page_height,
        idx_len,
        vec![input_page_ptr, input_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.output_chip.air,
        page_height,
        0,
        vec![output_page_ptr, output_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.range_checker.air,
        1 << decomp,
        0,
        vec![range_checker_ptr],
    );

    let partial_pk = keygen_builder.generate_partial_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2177, 5880, 51171, 3989, 12770],
    ];

    let x: Vec<u32> = vec![2177, 5880];

    let page_output =
        page_controller.gen_output(page.clone(), x.clone(), idx_len, page_width, Comp::Lte);

    index_scan_test(
        &engine,
        page,
        page_output,
        x,
        idx_len,
        data_len,
        limb_bits,
        decomp,
        &mut page_controller,
        &mut trace_builder,
        &partial_pk,
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_eq() {
    let bus_index: usize = 0;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: Vec<usize> = vec![16, 16];
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;
    let page_height = 1 << log_page_height;
    let page_width = 1 + idx_len + data_len;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits.clone(),
        decomp,
        Comp::Eq,
    );

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    let input_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let output_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let input_page_aux_ptr = keygen_builder.add_main_matrix(page_controller.input_chip.aux_width());
    let output_page_aux_ptr =
        keygen_builder.add_main_matrix(page_controller.output_chip.aux_width());
    let range_checker_ptr =
        keygen_builder.add_main_matrix(page_controller.range_checker.air_width());

    keygen_builder.add_partitioned_air(
        &page_controller.input_chip.air,
        page_height,
        idx_len,
        vec![input_page_ptr, input_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.output_chip.air,
        page_height,
        0,
        vec![output_page_ptr, output_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.range_checker.air,
        1 << decomp,
        0,
        vec![range_checker_ptr],
    );

    let partial_pk = keygen_builder.generate_partial_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];

    let x: Vec<u32> = vec![443, 376];

    let page_output =
        page_controller.gen_output(page.clone(), x.clone(), idx_len, page_width, Comp::Eq);

    index_scan_test(
        &engine,
        page,
        page_output,
        x,
        idx_len,
        data_len,
        limb_bits,
        decomp,
        &mut page_controller,
        &mut trace_builder,
        &partial_pk,
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_gte() {
    let bus_index: usize = 0;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: Vec<usize> = vec![16, 16];
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;
    let page_height = 1 << log_page_height;
    let page_width = 1 + idx_len + data_len;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits.clone(),
        decomp,
        Comp::Gte,
    );

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    let input_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let output_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let input_page_aux_ptr = keygen_builder.add_main_matrix(page_controller.input_chip.aux_width());
    let output_page_aux_ptr =
        keygen_builder.add_main_matrix(page_controller.output_chip.aux_width());
    let range_checker_ptr =
        keygen_builder.add_main_matrix(page_controller.range_checker.air_width());

    keygen_builder.add_partitioned_air(
        &page_controller.input_chip.air,
        page_height,
        idx_len,
        vec![input_page_ptr, input_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.output_chip.air,
        page_height,
        0,
        vec![output_page_ptr, output_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.range_checker.air,
        1 << decomp,
        0,
        vec![range_checker_ptr],
    );

    let partial_pk = keygen_builder.generate_partial_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let page: Vec<Vec<u32>> = vec![
        vec![1, 2177, 5880, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];

    let x: Vec<u32> = vec![2177, 5880];

    let page_output =
        page_controller.gen_output(page.clone(), x.clone(), idx_len, page_width, Comp::Gte);

    println!("{:?}", page_output);

    index_scan_test(
        &engine,
        page,
        page_output,
        x,
        idx_len,
        data_len,
        limb_bits,
        decomp,
        &mut page_controller,
        &mut trace_builder,
        &partial_pk,
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_gt() {
    let bus_index: usize = 0;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: Vec<usize> = vec![16, 16];
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;
    let page_height = 1 << log_page_height;
    let page_width = 1 + idx_len + data_len;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits.clone(),
        decomp,
        Comp::Gt,
    );

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    let input_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let output_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let input_page_aux_ptr = keygen_builder.add_main_matrix(page_controller.input_chip.aux_width());
    let output_page_aux_ptr =
        keygen_builder.add_main_matrix(page_controller.output_chip.aux_width());
    let range_checker_ptr =
        keygen_builder.add_main_matrix(page_controller.range_checker.air_width());

    keygen_builder.add_partitioned_air(
        &page_controller.input_chip.air,
        page_height,
        idx_len,
        vec![input_page_ptr, input_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.output_chip.air,
        page_height,
        0,
        vec![output_page_ptr, output_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.range_checker.air,
        1 << decomp,
        0,
        vec![range_checker_ptr],
    );

    let partial_pk = keygen_builder.generate_partial_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let page: Vec<Vec<u32>> = vec![
        vec![1, 2203, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];

    let x: Vec<u32> = vec![2177, 5880];

    let page_output =
        page_controller.gen_output(page.clone(), x.clone(), idx_len, page_width, Comp::Gt);

    index_scan_test(
        &engine,
        page,
        page_output,
        x,
        idx_len,
        data_len,
        limb_bits,
        decomp,
        &mut page_controller,
        &mut trace_builder,
        &partial_pk,
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_wrong_order() {
    let bus_index: usize = 0;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: Vec<usize> = vec![16, 16];
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;
    let page_height = 1 << log_page_height;
    let page_width = 1 + idx_len + data_len;

    let cmp = Comp::Lt;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits.clone(),
        decomp,
        cmp,
    );

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    let input_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let output_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let input_page_aux_ptr = keygen_builder.add_main_matrix(page_controller.input_chip.aux_width());
    let output_page_aux_ptr =
        keygen_builder.add_main_matrix(page_controller.output_chip.aux_width());
    let range_checker_ptr =
        keygen_builder.add_main_matrix(page_controller.range_checker.air_width());

    keygen_builder.add_partitioned_air(
        &page_controller.input_chip.air,
        page_height,
        idx_len,
        vec![input_page_ptr, input_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.output_chip.air,
        page_height,
        0,
        vec![output_page_ptr, output_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.range_checker.air,
        1 << decomp,
        0,
        vec![range_checker_ptr],
    );

    let partial_pk = keygen_builder.generate_partial_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = vec![
        vec![0, 0, 0, 0, 0, 0],
        vec![1, 443, 376, 22278, 13998, 58327],
    ];

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        index_scan_test(
            &engine,
            page,
            page_output,
            x,
            idx_len,
            data_len,
            limb_bits,
            decomp,
            &mut page_controller,
            &mut trace_builder,
            &partial_pk,
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_single_page_index_scan_unsorted() {
    let bus_index: usize = 0;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: Vec<usize> = vec![16, 16];
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;
    let page_height = 1 << log_page_height;
    let page_width = 1 + idx_len + data_len;

    let cmp = Comp::Lt;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits.clone(),
        decomp,
        cmp,
    );

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    let input_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let output_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let input_page_aux_ptr = keygen_builder.add_main_matrix(page_controller.input_chip.aux_width());
    let output_page_aux_ptr =
        keygen_builder.add_main_matrix(page_controller.output_chip.aux_width());
    let range_checker_ptr =
        keygen_builder.add_main_matrix(page_controller.range_checker.air_width());

    keygen_builder.add_partitioned_air(
        &page_controller.input_chip.air,
        page_height,
        idx_len,
        vec![input_page_ptr, input_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.output_chip.air,
        page_height,
        0,
        vec![output_page_ptr, output_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.range_checker.air,
        1 << decomp,
        0,
        vec![range_checker_ptr],
    );

    let partial_pk = keygen_builder.generate_partial_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let page: Vec<Vec<u32>> = vec![
        vec![1, 2883, 7769, 51171, 3989, 12770],
        vec![1, 443, 376, 22278, 13998, 58327],
    ];

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = vec![
        vec![0, 0, 0, 0, 0, 0],
        vec![1, 443, 376, 22278, 13998, 58327],
    ];

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        index_scan_test(
            &engine,
            page,
            page_output,
            x,
            idx_len,
            data_len,
            limb_bits,
            decomp,
            &mut page_controller,
            &mut trace_builder,
            &partial_pk,
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
