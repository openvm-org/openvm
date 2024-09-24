use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use ax_sdk::config::{
    self,
    baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
};

use super::{page_controller::PageController, page_index_scan_input::Comp};
use crate::common::page::Page;

const PAGE_BUS_INDEX: usize = 0;
const RANGE_BUS_INDEX: usize = 1;
const IDX_LEN: usize = 2;
const DATA_LEN: usize = 3;
const DECOMP: usize = 8;
const LIMB_BITS: usize = 16;

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
    let page_height = page.height();
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

    page_controller.set_up_keygen_builder(&mut keygen_builder, page_width, idx_len);

    let pk = keygen_builder.generate_pk();

    let proof = page_controller.prove(
        engine,
        &pk,
        trace_builder,
        input_prover_data,
        output_prover_data,
        x,
        idx_decomp,
    );
    let vk = pk.vk();

    page_controller.verify(engine, vk, &proof)
}

#[test]
fn test_single_page_index_scan_lt() {
    let cmp = Comp::Lt;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
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

    let expected_page_output = Page::from_2d_vec(
        &[
            vec![1, 443, 376, 22278, 13998, 58327],
            vec![0; 1 + IDX_LEN + DATA_LEN],
        ],
        IDX_LEN,
        DATA_LEN,
    );
    let page_output = page_controller.gen_output(page.clone(), x.clone(), PAGE_WIDTH, cmp);
    assert_eq!(expected_page_output, page_output);

    let engine = config::baby_bear_poseidon2::default_engine(27);

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

    let engine = config::baby_bear_poseidon2::default_engine(27);

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

    let engine = config::baby_bear_poseidon2::default_engine(27);

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

    let engine = config::baby_bear_poseidon2::default_engine(27);

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

    let engine = config::baby_bear_poseidon2::default_engine(27);

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

    let engine = config::baby_bear_poseidon2::default_engine(27);

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    disable_debug_builder();
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

    let engine = config::baby_bear_poseidon2::default_engine(27);

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    disable_debug_builder();
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

    let engine = config::baby_bear_poseidon2::default_engine(27);

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    disable_debug_builder();
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
