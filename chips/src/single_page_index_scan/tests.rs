use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver, USE_DEBUG_BUILDER},
    verifier::VerificationError,
};
use afs_test_utils::config::{
    self,
    baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
};

use crate::common::page::Page;

use super::{page_controller::PageController, page_index_scan_input::Comp};

#[allow(clippy::too_many_arguments)]
fn index_scan_test(
    engine: &BabyBearPoseidon2Engine,
    page: Page,
    page_output: Page,
    x: Vec<u32>,
    idx_len: usize,
    data_len: usize,
    idx_limb_bits: usize,
    idx_decomp: usize,
    page_controller: &mut PageController<BabyBearPoseidon2Config>,
    trace_builder: &mut TraceCommitmentBuilder<BabyBearPoseidon2Config>,
) -> Result<(), VerificationError> {
    let page_height = page.rows.len();
    assert!(page_height > 0);

    page_controller.load_page(
        page.clone(),
        page_output.clone(),
        x.clone(),
        idx_len,
        data_len,
        idx_limb_bits,
        idx_decomp,
        &mut trace_builder.committer,
    );

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);
    let page_width = 1 + idx_len + data_len;
    let page_height = page.rows.len();

    page_controller.set_up_keygen_builder(
        &mut keygen_builder,
        page_width,
        page_height,
        idx_len,
        idx_decomp,
    );

    let partial_pk = keygen_builder.generate_partial_pk();

    let proof = page_controller.prove(engine, &partial_pk, trace_builder, x.clone(), idx_decomp);
    let partial_vk = partial_pk.partial_vk();

    page_controller.verify(engine, partial_vk, proof, x.clone())
}

#[test]
fn test_single_page_index_scan_lt() {
    let page_bus_index: usize = 0;
    let range_bus_index: usize = 1;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: usize = 16;
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;
    let page_width = 1 + idx_len + data_len;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        page_bus_index,
        range_bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits,
        decomp,
        Comp::Lt,
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];
    let page = Page::from_2d_vec(&page, idx_len, data_len);

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = page_controller.gen_output(page.clone(), x.clone(), page_width, Comp::Lt);

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

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
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_lte() {
    let page_bus_index: usize = 0;
    let range_bus_index: usize = 1;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: usize = 16;
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;
    let page_width = 1 + idx_len + data_len;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        page_bus_index,
        range_bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits,
        decomp,
        Comp::Lte,
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2177, 5880, 51171, 3989, 12770],
    ];
    let page = Page::from_2d_vec(&page, idx_len, data_len);

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = page_controller.gen_output(page.clone(), x.clone(), page_width, Comp::Lte);

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

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
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_eq() {
    let page_bus_index: usize = 0;
    let range_bus_index: usize = 1;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: usize = 16;
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;
    let page_width = 1 + idx_len + data_len;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        page_bus_index,
        range_bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits,
        decomp,
        Comp::Eq,
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];
    let page = Page::from_2d_vec(&page, idx_len, data_len);

    let x: Vec<u32> = vec![443, 376];

    let page_output = page_controller.gen_output(page.clone(), x.clone(), page_width, Comp::Eq);

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

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
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_gte() {
    let page_bus_index: usize = 0;
    let range_bus_index: usize = 1;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: usize = 16;
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;
    let page_width = 1 + idx_len + data_len;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        page_bus_index,
        range_bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits,
        decomp,
        Comp::Gte,
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 2177, 5880, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];
    let page = Page::from_2d_vec(&page, idx_len, data_len);

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = page_controller.gen_output(page.clone(), x.clone(), page_width, Comp::Gte);

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

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
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_gt() {
    let page_bus_index: usize = 0;
    let range_bus_index: usize = 1;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: usize = 16;
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;
    let page_width = 1 + idx_len + data_len;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        page_bus_index,
        range_bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits,
        decomp,
        Comp::Gt,
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 2203, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];
    let page = Page::from_2d_vec(&page, idx_len, data_len);

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = page_controller.gen_output(page.clone(), x.clone(), page_width, Comp::Gt);

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

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
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_wrong_order() {
    let page_bus_index: usize = 0;
    let range_bus_index: usize = 1;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: usize = 16;
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;

    let cmp = Comp::Lt;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        page_bus_index,
        range_bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits,
        decomp,
        cmp,
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];
    let page = Page::from_2d_vec(&page, idx_len, data_len);

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = vec![
        vec![0, 0, 0, 0, 0, 0],
        vec![1, 443, 376, 22278, 13998, 58327],
    ];
    let page_output = Page::from_2d_vec(&page_output, idx_len, data_len);

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

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
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_single_page_index_scan_unsorted() {
    let page_bus_index: usize = 0;
    let range_bus_index: usize = 1;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: usize = 16;
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;

    let cmp = Comp::Lt;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        page_bus_index,
        range_bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits,
        decomp,
        cmp,
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 2883, 7769, 51171, 3989, 12770],
        vec![1, 443, 376, 22278, 13998, 58327],
    ];
    let page = Page::from_2d_vec(&page, idx_len, data_len);

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = vec![
        vec![0, 0, 0, 0, 0, 0],
        vec![1, 443, 376, 22278, 13998, 58327],
    ];
    let page_output = Page::from_2d_vec(&page_output, idx_len, data_len);

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

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
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_single_page_index_scan_wrong_answer() {
    let page_bus_index: usize = 0;
    let range_bus_index: usize = 1;
    let idx_len: usize = 2;
    let data_len: usize = 3;
    let decomp: usize = 8;
    let limb_bits: usize = 16;
    let range_max: u32 = 1 << decomp;

    let log_page_height = 1;

    let cmp = Comp::Lt;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        page_bus_index,
        range_bus_index,
        idx_len,
        data_len,
        range_max,
        limb_bits,
        decomp,
        cmp,
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 2883, 7769, 51171, 3989, 12770],
        vec![1, 443, 376, 22278, 13998, 58327],
    ];
    let page = Page::from_2d_vec(&page, idx_len, data_len);

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = vec![
        vec![1, 2883, 7769, 51171, 3989, 12770],
        vec![0, 0, 0, 0, 0, 0],
    ];
    let page_output = Page::from_2d_vec(&page_output, idx_len, data_len);

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(decomp));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

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
        ),
        Err(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it passed"
    );
}
