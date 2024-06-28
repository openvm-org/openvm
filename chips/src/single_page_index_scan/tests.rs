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

const PAGE_BUS_INDEX: usize = 0;
const RANGE_BUS_INDEX: usize = 1;
const IDX_LEN: usize = 2;
const DATA_LEN: usize = 3;
const DECOMP: usize = 8;
const LIMB_BITS: usize = 16;
const RANGE_MAX: u32 = 1 << DECOMP;

const LOG_PAGE_HEIGHT: usize = 1;
const PAGE_WIDTH: usize = 1 + IDX_LEN + DATA_LEN;

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

    let (input_prover_data, output_prover_data) = page_controller.load_page(
        page.clone(),
        page_output.clone(),
        None,
        None,
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

    let proof = page_controller.prove(
        engine,
        &partial_pk,
        trace_builder,
        input_prover_data,
        output_prover_data,
        x.clone(),
        idx_decomp,
    );
    let partial_vk = partial_pk.partial_vk();

    page_controller.verify(engine, partial_vk, proof, x.clone())
}

#[test]
fn test_single_page_index_scan_lt() {
    let cmp = Comp::Lt;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
        RANGE_MAX,
        LIMB_BITS,
        DECOMP,
        cmp.clone(),
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];
    let page = Page::from_2d_vec(&page, IDX_LEN, DATA_LEN);

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = page_controller.gen_output(page.clone(), x.clone(), PAGE_WIDTH, cmp);

    let engine = config::baby_bear_poseidon2::default_engine(LOG_PAGE_HEIGHT.max(DECOMP));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    index_scan_test(
        &engine,
        page,
        page_output,
        x,
        IDX_LEN,
        DATA_LEN,
        LIMB_BITS,
        DECOMP,
        &mut page_controller,
        &mut trace_builder,
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_lte() {
    let cmp = Comp::Lte;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
        RANGE_MAX,
        LIMB_BITS,
        DECOMP,
        cmp.clone(),
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2177, 5880, 51171, 3989, 12770],
    ];
    let page = Page::from_2d_vec(&page, IDX_LEN, DATA_LEN);

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = page_controller.gen_output(page.clone(), x.clone(), PAGE_WIDTH, cmp);

    let engine = config::baby_bear_poseidon2::default_engine(LOG_PAGE_HEIGHT.max(DECOMP));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    index_scan_test(
        &engine,
        page,
        page_output,
        x,
        IDX_LEN,
        DATA_LEN,
        LIMB_BITS,
        DECOMP,
        &mut page_controller,
        &mut trace_builder,
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_eq() {
    let cmp = Comp::Eq;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
        RANGE_MAX,
        LIMB_BITS,
        DECOMP,
        cmp.clone(),
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];
    let page = Page::from_2d_vec(&page, IDX_LEN, DATA_LEN);

    let x: Vec<u32> = vec![443, 376];

    let page_output = page_controller.gen_output(page.clone(), x.clone(), PAGE_WIDTH, cmp);

    let engine = config::baby_bear_poseidon2::default_engine(LOG_PAGE_HEIGHT.max(DECOMP));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    index_scan_test(
        &engine,
        page,
        page_output,
        x,
        IDX_LEN,
        DATA_LEN,
        LIMB_BITS,
        DECOMP,
        &mut page_controller,
        &mut trace_builder,
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_gte() {
    let cmp = Comp::Gte;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
        RANGE_MAX,
        LIMB_BITS,
        DECOMP,
        cmp.clone(),
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 2177, 5880, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];
    let page = Page::from_2d_vec(&page, IDX_LEN, DATA_LEN);

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = page_controller.gen_output(page.clone(), x.clone(), PAGE_WIDTH, cmp);

    let engine = config::baby_bear_poseidon2::default_engine(LOG_PAGE_HEIGHT.max(DECOMP));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    index_scan_test(
        &engine,
        page,
        page_output,
        x,
        IDX_LEN,
        DATA_LEN,
        LIMB_BITS,
        DECOMP,
        &mut page_controller,
        &mut trace_builder,
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_gt() {
    let cmp = Comp::Gt;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
        RANGE_MAX,
        LIMB_BITS,
        DECOMP,
        cmp.clone(),
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 2203, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];
    let page = Page::from_2d_vec(&page, IDX_LEN, DATA_LEN);

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = page_controller.gen_output(page.clone(), x.clone(), PAGE_WIDTH, cmp);

    let engine = config::baby_bear_poseidon2::default_engine(LOG_PAGE_HEIGHT.max(DECOMP));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    index_scan_test(
        &engine,
        page,
        page_output,
        x,
        IDX_LEN,
        DATA_LEN,
        LIMB_BITS,
        DECOMP,
        &mut page_controller,
        &mut trace_builder,
    )
    .expect("Verification failed");
}

#[test]
fn test_single_page_index_scan_wrong_order() {
    let cmp = Comp::Lt;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
        RANGE_MAX,
        LIMB_BITS,
        DECOMP,
        cmp,
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 22278, 13998, 58327],
        vec![1, 2883, 7769, 51171, 3989, 12770],
    ];
    let page = Page::from_2d_vec(&page, IDX_LEN, DATA_LEN);

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = vec![
        vec![0, 0, 0, 0, 0, 0],
        vec![1, 443, 376, 22278, 13998, 58327],
    ];
    let page_output = Page::from_2d_vec(&page_output, IDX_LEN, DATA_LEN);

    let engine = config::baby_bear_poseidon2::default_engine(LOG_PAGE_HEIGHT.max(DECOMP));

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
            IDX_LEN,
            DATA_LEN,
            LIMB_BITS,
            DECOMP,
            &mut page_controller,
            &mut trace_builder,
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_single_page_index_scan_unsorted() {
    let cmp = Comp::Lt;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
        RANGE_MAX,
        LIMB_BITS,
        DECOMP,
        cmp,
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 2883, 7769, 51171, 3989, 12770],
        vec![1, 443, 376, 22278, 13998, 58327],
    ];
    let page = Page::from_2d_vec(&page, IDX_LEN, DATA_LEN);

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = vec![
        vec![0, 0, 0, 0, 0, 0],
        vec![1, 443, 376, 22278, 13998, 58327],
    ];
    let page_output = Page::from_2d_vec(&page_output, IDX_LEN, DATA_LEN);

    let engine = config::baby_bear_poseidon2::default_engine(LOG_PAGE_HEIGHT.max(DECOMP));

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
            IDX_LEN,
            DATA_LEN,
            LIMB_BITS,
            DECOMP,
            &mut page_controller,
            &mut trace_builder,
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_single_page_index_scan_wrong_answer() {
    let cmp = Comp::Lt;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
        RANGE_MAX,
        LIMB_BITS,
        DECOMP,
        cmp,
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 2883, 7769, 51171, 3989, 12770],
        vec![1, 443, 376, 22278, 13998, 58327],
    ];
    let page = Page::from_2d_vec(&page, IDX_LEN, DATA_LEN);

    let x: Vec<u32> = vec![2177, 5880];

    let page_output = vec![
        vec![1, 2883, 7769, 51171, 3989, 12770],
        vec![0, 0, 0, 0, 0, 0],
    ];
    let page_output = Page::from_2d_vec(&page_output, IDX_LEN, DATA_LEN);

    let engine = config::baby_bear_poseidon2::default_engine(LOG_PAGE_HEIGHT.max(DECOMP));

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
            IDX_LEN,
            DATA_LEN,
            LIMB_BITS,
            DECOMP,
            &mut page_controller,
            &mut trace_builder,
        ),
        Err(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it passed"
    );
}
