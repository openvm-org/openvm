use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
    verifier::VerificationError,
};
use afs_test_utils::config::{
    self,
    baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
};

use super::page_controller::PageController;
use crate::common::{comp::Comp, page::Page};

const PAGE_BUS_INDEX: usize = 0;
const RANGE_BUS_INDEX: usize = 1;
const IDX_LEN: usize = 2;
const DATA_LEN: usize = 3;
const PAGE_WIDTH: usize = 1 + IDX_LEN + DATA_LEN;
const DECOMP: usize = 8;
const LIMB_BITS: usize = 16;
const RANGE_MAX: u32 = 1 << DECOMP;

const LOG_PAGE_HEIGHT: usize = 2;

#[test]
pub fn test_gen_output() {
    let page_controller = PageController::<BabyBearPoseidon2Config>::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
        2,
        4,
        RANGE_MAX,
        LIMB_BITS,
        DECOMP,
        Comp::Lt,
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 2278, 1399, 58327],
        vec![1, 2883, 7269, 4171, 3989, 12770],
        vec![1, 4826, 7969, 51171, 989, 12770],
        vec![1, 6588, 8069, 82142, 500, 12770],
    ];
    let page = Page::from_2d_vec(&page, IDX_LEN, DATA_LEN);

    let expected_output = vec![
        vec![1, 443, 376, 2278, 1399, 58327],
        vec![1, 2883, 7269, 4171, 3989, 12770],
        vec![0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0],
    ];
    let expected_output = Page::from_2d_vec(&expected_output, IDX_LEN, DATA_LEN);

    let x: Vec<u32> = vec![32278, 520];

    let page_output = page_controller.gen_output(page.clone(), x.clone(), Comp::Lt);
    assert_eq!(expected_output, page_output);
}

#[allow(clippy::too_many_arguments)]
fn filter_test(
    engine: &BabyBearPoseidon2Engine,
    page: Page,
    page_output: Page,
    x: Vec<u32>,
    idx_len: usize,
    data_len: usize,
    start_col: usize,
    end_col: usize,
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
        start_col,
        end_col,
        idx_limb_bits,
        idx_decomp,
        &mut trace_builder.committer,
    );

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);
    let page_width = 1 + idx_len + data_len;
    let select_len = end_col - start_col;

    page_controller.set_up_keygen_builder(&mut keygen_builder, page_width, select_len);

    let pk = keygen_builder.generate_pk();

    let proof = page_controller.prove(
        engine,
        &pk,
        trace_builder,
        input_prover_data,
        output_prover_data,
        x.clone(),
        idx_decomp,
    );
    let vk = pk.vk();

    page_controller.verify(engine, vk, proof, x.clone())
}

#[test]
fn test_filter_lt() {
    let cmp = Comp::Lt;

    let start_col = 2;
    let end_col = 4;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
        start_col,
        end_col,
        RANGE_MAX,
        LIMB_BITS,
        DECOMP,
        cmp.clone(),
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 2278, 1399, 58327],
        vec![1, 2883, 7269, 4171, 3989, 12770],
        vec![1, 4826, 7969, 51171, 989, 12770],
        vec![1, 6588, 8069, 82142, 500, 12770],
    ];
    let page = Page::from_2d_vec(&page, IDX_LEN, DATA_LEN);

    let x: Vec<u32> = vec![32278, 520];

    let expected_page_output = Page::from_2d_vec(
        &[
            vec![1, 443, 376, 2278, 1399, 58327],
            vec![1, 2883, 7269, 4171, 3989, 12770],
            vec![0; PAGE_WIDTH],
            vec![0; PAGE_WIDTH],
        ],
        IDX_LEN,
        DATA_LEN,
    );
    let page_output = page_controller.gen_output(page.clone(), x.clone(), cmp);
    assert_eq!(expected_page_output, page_output);

    let engine = config::baby_bear_poseidon2::default_engine(LOG_PAGE_HEIGHT.max(DECOMP));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    filter_test(
        &engine,
        page,
        page_output,
        x,
        IDX_LEN,
        DATA_LEN,
        start_col,
        end_col,
        LIMB_BITS,
        DECOMP,
        &mut page_controller,
        &mut trace_builder,
    )
    .expect("Verification failed");
}

#[test]
fn test_filter_gt() {
    let cmp = Comp::Gt;

    let start_col = 1;
    let end_col = 4;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
        start_col,
        end_col,
        RANGE_MAX,
        LIMB_BITS,
        DECOMP,
        cmp.clone(),
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 2278, 1399, 58327],
        vec![1, 2883, 7269, 4171, 3989, 12770],
        vec![1, 4826, 7969, 51171, 989, 12770],
        vec![1, 6588, 8069, 82142, 500, 12770],
    ];
    let page = Page::from_2d_vec(&page, IDX_LEN, DATA_LEN);

    let x = vec![7500, 32278, 10];

    let expected_page_output = Page::from_2d_vec(
        &[
            vec![1, 4826, 7969, 51171, 989, 12770],
            vec![1, 6588, 8069, 82142, 500, 12770],
            vec![0; PAGE_WIDTH],
            vec![0; PAGE_WIDTH],
        ],
        IDX_LEN,
        DATA_LEN,
    );
    let page_output = page_controller.gen_output(page.clone(), x.clone(), cmp);
    assert_eq!(expected_page_output, page_output);

    let engine = config::baby_bear_poseidon2::default_engine(LOG_PAGE_HEIGHT.max(DECOMP));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    filter_test(
        &engine,
        page,
        page_output,
        x,
        IDX_LEN,
        DATA_LEN,
        start_col,
        end_col,
        LIMB_BITS,
        DECOMP,
        &mut page_controller,
        &mut trace_builder,
    )
    .expect("Verification failed");
}

#[test]
fn test_filter_lte() {
    let cmp = Comp::Lte;

    let start_col = 2;
    let end_col = 4;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
        start_col,
        end_col,
        RANGE_MAX,
        LIMB_BITS,
        DECOMP,
        cmp.clone(),
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 2278, 1399, 58327],
        vec![1, 2883, 7269, 4171, 3989, 12770],
        vec![1, 4826, 7969, 51171, 989, 12770],
        vec![1, 6588, 8069, 82142, 500, 12770],
    ];
    let page = Page::from_2d_vec(&page, IDX_LEN, DATA_LEN);

    let x: Vec<u32> = vec![51171, 989];

    let expected_page_output = Page::from_2d_vec(
        &[
            vec![1, 443, 376, 2278, 1399, 58327],
            vec![1, 2883, 7269, 4171, 3989, 12770],
            vec![1, 4826, 7969, 51171, 989, 12770],
            vec![0; PAGE_WIDTH],
        ],
        IDX_LEN,
        DATA_LEN,
    );
    let page_output = page_controller.gen_output(page.clone(), x.clone(), cmp);
    assert_eq!(expected_page_output, page_output);

    let engine = config::baby_bear_poseidon2::default_engine(LOG_PAGE_HEIGHT.max(DECOMP));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    filter_test(
        &engine,
        page,
        page_output,
        x,
        IDX_LEN,
        DATA_LEN,
        start_col,
        end_col,
        LIMB_BITS,
        DECOMP,
        &mut page_controller,
        &mut trace_builder,
    )
    .expect("Verification failed");
}

#[test]
fn test_filter_gte() {
    let cmp = Comp::Gte;

    let start_col = 2;
    let end_col = 4;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
        start_col,
        end_col,
        RANGE_MAX,
        LIMB_BITS,
        DECOMP,
        cmp.clone(),
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 2278, 1399, 58327],
        vec![1, 2883, 7269, 4171, 3989, 12770],
        vec![1, 4826, 7969, 51171, 989, 12770],
        vec![1, 6588, 8069, 82142, 500, 12770],
    ];
    let page = Page::from_2d_vec(&page, IDX_LEN, DATA_LEN);

    let x: Vec<u32> = vec![82142, 500];

    let expected_page_output = Page::from_2d_vec(
        &[
            vec![1, 6588, 8069, 82142, 500, 12770],
            vec![0; PAGE_WIDTH],
            vec![0; PAGE_WIDTH],
            vec![0; PAGE_WIDTH],
        ],
        IDX_LEN,
        DATA_LEN,
    );
    let page_output = page_controller.gen_output(page.clone(), x.clone(), cmp);
    assert_eq!(expected_page_output, page_output);

    let engine = config::baby_bear_poseidon2::default_engine(LOG_PAGE_HEIGHT.max(DECOMP));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    filter_test(
        &engine,
        page,
        page_output,
        x,
        IDX_LEN,
        DATA_LEN,
        start_col,
        end_col,
        LIMB_BITS,
        DECOMP,
        &mut page_controller,
        &mut trace_builder,
    )
    .expect("Verification failed");
}

#[test]
fn test_filter_eq() {
    let cmp = Comp::Eq;

    let start_col = 2;
    let end_col = 5;

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        IDX_LEN,
        DATA_LEN,
        start_col,
        end_col,
        RANGE_MAX,
        LIMB_BITS,
        DECOMP,
        cmp.clone(),
    );

    let page: Vec<Vec<u32>> = vec![
        vec![1, 443, 376, 2278, 1399, 58327],
        vec![1, 2883, 7269, 4171, 3989, 12770],
        vec![1, 4826, 7969, 51171, 989, 12770],
        vec![1, 6588, 8069, 82142, 500, 12770],
    ];
    let page = Page::from_2d_vec(&page, IDX_LEN, DATA_LEN);

    let x: Vec<u32> = vec![4171, 3989, 12770];

    let expected_page_output = Page::from_2d_vec(
        &[
            vec![1, 2883, 7269, 4171, 3989, 12770],
            vec![0; PAGE_WIDTH],
            vec![0; PAGE_WIDTH],
            vec![0; PAGE_WIDTH],
        ],
        IDX_LEN,
        DATA_LEN,
    );
    let page_output = page_controller.gen_output(page.clone(), x.clone(), cmp);
    assert_eq!(expected_page_output, page_output);

    let engine = config::baby_bear_poseidon2::default_engine(LOG_PAGE_HEIGHT.max(DECOMP));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    filter_test(
        &engine,
        page,
        page_output,
        x,
        IDX_LEN,
        DATA_LEN,
        start_col,
        end_col,
        LIMB_BITS,
        DECOMP,
        &mut page_controller,
        &mut trace_builder,
    )
    .expect("Verification failed");
}
