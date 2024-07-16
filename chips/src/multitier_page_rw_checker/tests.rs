use std::collections::HashMap;

use std::iter;
use std::sync::Arc;

use afs_stark_backend::prover::trace::TraceCommitter;
use afs_stark_backend::rap::AnyRap;
use afs_stark_backend::verifier::VerificationError;
use afs_stark_backend::{
    keygen::{types::MultiStarkPartialProvingKey, MultiStarkKeygenBuilder},
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
};
use afs_test_utils::config::{
    self,
    baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
};
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use afs_test_utils::{engine::StarkEngine, utils::create_seeded_rng};

use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

use crate::multitier_page_rw_checker::page_controller::{
    MyLessThanTupleParams, PageController, PageTreeParams,
};
use crate::page_btree::{PageBTree, PageBTreePages};
use crate::page_rw_checker::page_controller::{OpType, Operation};
use crate::range_gate::RangeCheckerGateChip;

use super::page_controller;

pub const BABYBEAR_COMMITMENT_LEN: usize = 8;
pub const DECOMP_BITS: usize = 6;
const MAX_LIMB_VAL: u32 = 1 << 20;
const MAX_VAL: u32 = 0x78000001 / 2;

type Val = BabyBear;

#[test]
fn multitier_page_rw_no_new_keys() {
    multitier_page_rw_test(generate_no_new_keys, false, 4);
}

#[test]
fn multitier_page_rw_new_keys() {
    multitier_page_rw_test(generate_new_keys, false, 4);
}

#[test]
fn multitier_page_rw_mixed_ops() {
    multitier_page_rw_test(generate_mixed_ops, false, 4);
}

#[test]
fn multitier_page_rw_mixed_ops_empty_start() {
    multitier_page_rw_test(generate_mixed_ops_empty_start, false, 4);
}

#[test]
fn multitier_page_rw_mixed_ops_remove_first_leaf() {
    multitier_page_rw_test(generate_mixed_ops_remove_first_leaf, false, 4);
}

// These next tests require the large tree to be on disk
// Run pagebtree::tests::make_a_large_tree to do this
#[ignore]
#[test]
fn multitier_page_rw_large_tree_no_new_keys() {
    multitier_page_rw_test(generate_large_tree_no_new_keys, false, 5);
}

#[ignore]
#[test]
fn multitier_page_rw_large_tree_new_keys() {
    multitier_page_rw_test(generate_large_tree_new_keys, false, 5);
}

fn multitier_page_rw_test<F>(generate_inputs: F, should_fail: bool, log_page_height: usize)
where
    F: Fn(
        usize,
        usize,
        usize,
        usize,
        usize,
        &mut TraceCommitter<BabyBearPoseidon2Config>,
    ) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>),
{
    const DATA_BUS_INDEX: usize = 0;
    const INTERNAL_DATA_BUS_INDEX: usize = 1;
    const LT_BUS_INDEX: usize = 2;
    const INIT_PATH_BUS: usize = 3;
    const FINAL_PATH_BUS: usize = 4;
    const OPS_BUS_INDEX: usize = 5;

    let limb_bits = 20;

    let log_num_ops = 3;

    let page_height = 1 << log_page_height;
    let num_ops: usize = 1 << log_num_ops;

    let trace_degree = num_ops * 8;

    let idx_len = 2;
    let data_len = 3;

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(3 + log_num_ops));
    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let init_param = PageTreeParams {
        path_bus_index: INIT_PATH_BUS,
        leaf_cap: 8,
        internal_cap: 24,
        leaf_page_height: page_height,
        internal_page_height: page_height,
    };

    let final_param = PageTreeParams {
        path_bus_index: FINAL_PATH_BUS,
        leaf_cap: 8,
        internal_cap: 24,
        leaf_page_height: page_height,
        internal_page_height: page_height,
    };

    let less_than_tuple_param = MyLessThanTupleParams {
        limb_bits: 20,
        decomp: DECOMP_BITS,
    };

    let range_checker = Arc::new(RangeCheckerGateChip::new(LT_BUS_INDEX, 1 << DECOMP_BITS));

    let mut page_controller: PageController<BABYBEAR_COMMITMENT_LEN> =
        PageController::new::<BabyBearPoseidon2Config>(
            DATA_BUS_INDEX,
            INTERNAL_DATA_BUS_INDEX,
            OPS_BUS_INDEX,
            LT_BUS_INDEX,
            idx_len,
            data_len,
            init_param.clone(),
            final_param.clone(),
            less_than_tuple_param,
            range_checker,
        );
    let ops_sender = DummyInteractionAir::new(idx_len + data_len + 2, true, OPS_BUS_INDEX);
    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    let mut init_leaf_data_ptrs = vec![];
    let mut init_leaf_main_ptrs = vec![];

    let mut init_internal_data_ptrs = vec![];
    let mut init_internal_main_ptrs = vec![];

    let mut final_leaf_data_ptrs = vec![];
    let mut final_leaf_main_ptrs = vec![];

    let mut final_internal_data_ptrs = vec![];
    let mut final_internal_main_ptrs = vec![];

    for _ in 0..init_param.leaf_cap {
        init_leaf_data_ptrs.push(keygen_builder.add_cached_main_matrix(1 + idx_len + data_len));
    }

    for _ in 0..init_param.internal_cap {
        init_internal_data_ptrs
            .push(keygen_builder.add_cached_main_matrix(2 + 2 * idx_len + BABYBEAR_COMMITMENT_LEN));
    }

    for _ in 0..final_param.leaf_cap {
        final_leaf_data_ptrs.push(keygen_builder.add_cached_main_matrix(1 + idx_len + data_len));
    }

    for _ in 0..final_param.internal_cap {
        final_internal_data_ptrs
            .push(keygen_builder.add_cached_main_matrix(2 + 2 * idx_len + BABYBEAR_COMMITMENT_LEN));
    }

    for _ in 0..init_param.leaf_cap {
        init_leaf_main_ptrs.push(keygen_builder.add_main_matrix(
            page_controller.init_leaf_chips[0].air_width() - 1 - idx_len - data_len,
        ));
    }

    for _ in 0..init_param.internal_cap {
        init_internal_main_ptrs.push(keygen_builder.add_main_matrix(
            page_controller.init_internal_chips[0].air_width()
                - 2
                - 2 * idx_len
                - BABYBEAR_COMMITMENT_LEN,
        ));
    }

    for _ in 0..final_param.leaf_cap {
        final_leaf_main_ptrs.push(keygen_builder.add_main_matrix(
            page_controller.final_leaf_chips[0].air_width() - 1 - idx_len - data_len,
        ));
    }

    for _ in 0..final_param.internal_cap {
        final_internal_main_ptrs.push(keygen_builder.add_main_matrix(
            page_controller.final_internal_chips[0].air_width()
                - 2
                - 2 * idx_len
                - BABYBEAR_COMMITMENT_LEN,
        ));
    }

    let ops_ptr = keygen_builder.add_main_matrix(page_controller.offline_checker.air_width());

    let init_root_ptr =
        keygen_builder.add_main_matrix(page_controller.init_root_signal.air_width());
    let final_root_ptr =
        keygen_builder.add_main_matrix(page_controller.final_root_signal.air_width());

    for i in 0..init_param.leaf_cap {
        keygen_builder.add_partitioned_air(
            &page_controller.init_leaf_chips[i],
            BABYBEAR_COMMITMENT_LEN,
            vec![init_leaf_data_ptrs[i], init_leaf_main_ptrs[i]],
        );
    }

    for i in 0..init_param.internal_cap {
        keygen_builder.add_partitioned_air(
            &page_controller.init_internal_chips[i],
            BABYBEAR_COMMITMENT_LEN,
            vec![init_internal_data_ptrs[i], init_internal_main_ptrs[i]],
        );
    }

    for i in 0..final_param.leaf_cap {
        keygen_builder.add_partitioned_air(
            &page_controller.final_leaf_chips[i],
            BABYBEAR_COMMITMENT_LEN,
            vec![final_leaf_data_ptrs[i], final_leaf_main_ptrs[i]],
        );
    }

    for i in 0..final_param.internal_cap {
        keygen_builder.add_partitioned_air(
            &page_controller.final_internal_chips[i],
            BABYBEAR_COMMITMENT_LEN,
            vec![final_internal_data_ptrs[i], final_internal_main_ptrs[i]],
        );
    }

    keygen_builder.add_partitioned_air(&page_controller.offline_checker, 0, vec![ops_ptr]);

    keygen_builder.add_partitioned_air(
        &page_controller.init_root_signal,
        BABYBEAR_COMMITMENT_LEN,
        vec![init_root_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.final_root_signal,
        BABYBEAR_COMMITMENT_LEN,
        vec![final_root_ptr],
    );

    keygen_builder.add_air(&page_controller.range_checker.air, 0);

    keygen_builder.add_air(&ops_sender, 0);

    let partial_pk = keygen_builder.generate_partial_pk();
    let (init_pages, init_root_is_leaf, final_pages, final_root_is_leaf, ops) = generate_inputs(
        idx_len,
        data_len,
        page_height,
        limb_bits,
        num_ops,
        &mut trace_builder.committer,
    );
    let res = load_page_test(
        &engine,
        init_pages.leaf_pages.clone(),
        init_pages.internal_pages.clone(),
        init_root_is_leaf,
        0,
        final_pages.leaf_pages.clone(),
        final_pages.internal_pages.clone(),
        final_root_is_leaf,
        0,
        &ops,
        num_ops,
        &ops_sender,
        &mut page_controller,
        &mut trace_builder,
        &partial_pk,
        trace_degree,
    );
    assert!(should_fail == res.is_err());
    if !should_fail {
        res.unwrap();
    }
}

#[allow(clippy::too_many_arguments)]
fn load_page_test(
    engine: &BabyBearPoseidon2Engine,
    init_leaf_pages: Vec<Vec<Vec<u32>>>,
    init_internal_pages: Vec<Vec<Vec<u32>>>,
    init_root_is_leaf: bool,
    init_root_idx: usize,
    final_leaf_pages: Vec<Vec<Vec<u32>>>,
    final_internal_pages: Vec<Vec<Vec<u32>>>,
    final_root_is_leaf: bool,
    final_root_idx: usize,
    ops: &[Operation],
    num_ops: usize,
    ops_sender: &DummyInteractionAir,
    page_controller: &mut page_controller::PageController<BABYBEAR_COMMITMENT_LEN>,
    trace_builder: &mut TraceCommitmentBuilder<BabyBearPoseidon2Config>,
    partial_pk: &MultiStarkPartialProvingKey<BabyBearPoseidon2Config>,
    trace_degree: usize,
) -> Result<(), VerificationError> {
    page_controller.range_checker.clear();
    let (data_trace, main_trace, commits, mut prover_data) = page_controller.load_page_and_ops(
        init_leaf_pages,
        init_internal_pages,
        init_root_is_leaf,
        init_root_idx,
        final_leaf_pages,
        final_internal_pages,
        final_root_is_leaf,
        final_root_idx,
        ops.to_vec(),
        trace_degree,
        &mut trace_builder.committer,
    );
    let offline_checker_trace = main_trace.offline_checker_trace.clone();
    let init_root = main_trace.init_root_signal_trace.clone();
    let final_root = main_trace.final_root_signal_trace.clone();
    let range_trace = page_controller.range_checker.generate_trace();
    let ops_sender_trace = RowMajorMatrix::new(
        ops.iter()
            .flat_map(|op| {
                iter::once(Val::one())
                    .chain(iter::once(Val::from_canonical_usize(op.clk)))
                    .chain(op.idx.iter().map(|x| Val::from_canonical_u32(*x)))
                    .chain(op.data.iter().map(|x| Val::from_canonical_u32(*x)))
                    .chain(iter::once(Val::from_canonical_u8(op.op_type as u8)))
            })
            .chain(
                iter::repeat_with(|| iter::repeat(Val::zero()).take(1 + ops_sender.field_width()))
                    .take(num_ops - ops.len())
                    .flatten(),
            )
            .collect(),
        1 + ops_sender.field_width(),
    );
    trace_builder.clear();

    for trace in data_trace.init_leaf_chip_traces.iter() {
        trace_builder.load_cached_trace(trace.clone(), prover_data.init_leaf_page.remove(0));
    }

    for trace in data_trace.init_internal_chip_traces.iter() {
        trace_builder.load_cached_trace(trace.clone(), prover_data.init_internal_page.remove(0));
    }

    for trace in data_trace.final_leaf_chip_traces.iter() {
        trace_builder.load_cached_trace(trace.clone(), prover_data.final_leaf_page.remove(0));
    }

    for trace in data_trace.final_internal_chip_traces.iter() {
        trace_builder.load_cached_trace(trace.clone(), prover_data.final_internal_page.remove(0));
    }

    for trace in main_trace.init_leaf_chip_main_traces.iter() {
        trace_builder.load_trace(trace.clone());
    }

    for trace in main_trace.init_internal_chip_main_traces.iter() {
        trace_builder.load_trace(trace.clone());
    }

    for trace in main_trace.final_leaf_chip_main_traces.iter() {
        trace_builder.load_trace(trace.clone());
    }

    for trace in main_trace.final_internal_chip_main_traces.iter() {
        trace_builder.load_trace(trace.clone());
    }

    trace_builder.load_trace(offline_checker_trace.clone());
    trace_builder.load_trace(init_root);
    trace_builder.load_trace(final_root);
    trace_builder.load_trace(range_trace);
    trace_builder.load_trace(ops_sender_trace);
    trace_builder.commit_current();

    let mut airs: Vec<&dyn AnyRap<BabyBearPoseidon2Config>> = vec![];
    for chip in &page_controller.init_leaf_chips {
        airs.push(chip);
    }
    for chip in &page_controller.init_internal_chips {
        airs.push(chip);
    }
    for chip in &page_controller.final_leaf_chips {
        airs.push(chip);
    }
    for chip in &page_controller.final_internal_chips {
        airs.push(chip);
    }
    airs.push(&page_controller.offline_checker);
    airs.push(&page_controller.init_root_signal);
    airs.push(&page_controller.final_root_signal);
    airs.push(&page_controller.range_checker.air);
    airs.push(ops_sender);
    let partial_vk = partial_pk.partial_vk();
    let main_trace_data = trace_builder.view(&partial_vk, airs.clone());

    let mut pis = vec![];
    for c in commits.init_leaf_page_commitments {
        let c: [BabyBear; BABYBEAR_COMMITMENT_LEN] = c.into();
        pis.push(c.to_vec());
    }
    for c in commits.init_internal_page_commitments {
        let c: [BabyBear; BABYBEAR_COMMITMENT_LEN] = c.into();
        pis.push(c.to_vec());
    }
    for c in commits.final_leaf_page_commitments {
        let c: [BabyBear; BABYBEAR_COMMITMENT_LEN] = c.into();
        pis.push(c.to_vec());
    }
    for c in commits.final_internal_page_commitments {
        let c: [BabyBear; BABYBEAR_COMMITMENT_LEN] = c.into();
        pis.push(c.to_vec());
    }
    pis.push(vec![]);
    {
        let c: [BabyBear; BABYBEAR_COMMITMENT_LEN] = commits.init_root_commitment.into();
        pis.push(c.to_vec());
    }
    {
        let c: [BabyBear; BABYBEAR_COMMITMENT_LEN] = commits.final_root_commitment.into();
        pis.push(c.to_vec());
    }
    pis.push(vec![]);
    pis.push(vec![]);
    let prover = engine.prover();
    let verifier = engine.verifier();

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, partial_pk, main_trace_data, &pis);

    let mut challenger = engine.new_challenger();
    verifier.verify(&mut challenger, partial_vk, airs, proof, &pis)
}

#[allow(clippy::type_complexity)]
fn generate_init_tree_and_clks(
    idx_len: usize,
    data_len: usize,
    page_height: usize,
    limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (
    PageBTree<BABYBEAR_COMMITMENT_LEN>,
    Vec<usize>,
    PageBTreePages,
    HashMap<Vec<u32>, Vec<u32>>,
) {
    let mut rng = create_seeded_rng();
    let mut btree = PageBTree::<BABYBEAR_COMMITMENT_LEN>::new(
        limb_bits,
        idx_len,
        data_len,
        page_height,
        page_height,
    );
    // Generating a random page with distinct indices
    let mut idx_data_map = HashMap::new();
    for _ in 0..4 * page_height {
        let mut idx;
        loop {
            idx = (0..idx_len)
                .map(|_| rng.gen::<u32>() % MAX_LIMB_VAL)
                .collect::<Vec<u32>>();
            if !idx_data_map.contains_key(&idx) {
                break;
            }
        }

        let data: Vec<u32> = (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect();
        idx_data_map.insert(idx.clone(), data.clone());
        btree.update(&idx, &data);
    }
    let init_pages = btree.gen_all_trace(committer);
    // check for correctness
    for (idx, _) in idx_data_map.iter() {
        btree.search(idx).unwrap();
    }
    let mut clks: Vec<usize> = (0..num_ops)
        .map(|_| rng.gen::<usize>() % (MAX_VAL as usize))
        .collect();
    clks.sort();
    (btree, clks, init_pages, idx_data_map)
}

// generates a test case where we add no new keys to the original tree
fn generate_no_new_keys(
    idx_len: usize,
    data_len: usize,
    page_height: usize,
    limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>) {
    let (mut btree, clks, init_pages, mut idx_data_map) = generate_init_tree_and_clks(
        idx_len,
        data_len,
        page_height,
        limb_bits,
        num_ops,
        committer,
    );
    let mut rng = create_seeded_rng();
    let mut ops: Vec<Operation> = vec![];
    for clk in clks {
        let idx = idx_data_map
            .iter()
            .nth(rng.gen::<usize>() % idx_data_map.len())
            .unwrap()
            .0
            .to_vec();

        let op_type = {
            if rng.gen::<bool>() {
                OpType::Read
            } else {
                OpType::Write
            }
        };

        let data = {
            if op_type == OpType::Read {
                idx_data_map[&idx].to_vec()
            } else {
                (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect()
            }
        };

        if op_type == OpType::Write {
            idx_data_map.insert(idx.clone(), data.clone());
            btree.update(&idx, &data);
        }

        ops.push(Operation::new(clk, idx, data, op_type));
    }
    let final_pages = btree.gen_all_trace(committer);
    (init_pages, false, final_pages, false, ops)
}

// generates a test case where we add new keys to the original tree
fn generate_new_keys(
    idx_len: usize,
    data_len: usize,
    page_height: usize,
    limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>) {
    let (mut btree, clks, init_pages, mut idx_data_map) = generate_init_tree_and_clks(
        idx_len,
        data_len,
        page_height,
        limb_bits,
        num_ops,
        committer,
    );
    let mut rng = create_seeded_rng();
    let mut ops: Vec<Operation> = vec![];
    for clk in clks {
        let mut idx;
        loop {
            idx = (0..idx_len)
                .map(|_| rng.gen::<u32>() % MAX_LIMB_VAL)
                .collect::<Vec<u32>>();
            if !idx_data_map.contains_key(&idx) {
                break;
            }
        }

        let data: Vec<u32> = (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect();
        idx_data_map.insert(idx.clone(), data.clone());
        btree.update(&idx, &data);
        ops.push(Operation::new(clk, idx, data, OpType::Write));
    }
    let final_pages = btree.gen_all_trace(committer);
    (init_pages, false, final_pages, false, ops)
}

// generates a test case where we do all kinds of ops to the original tree
fn generate_mixed_ops(
    idx_len: usize,
    data_len: usize,
    page_height: usize,
    limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>) {
    let (mut btree, clks, init_pages, mut idx_data_map) = generate_init_tree_and_clks(
        idx_len,
        data_len,
        page_height,
        limb_bits,
        num_ops,
        committer,
    );
    let mut rng = create_seeded_rng();
    let mut ops: Vec<Operation> = vec![];
    for clk in clks {
        if rng.gen::<bool>() {
            let mut idx;
            loop {
                idx = (0..idx_len)
                    .map(|_| rng.gen::<u32>() % MAX_LIMB_VAL)
                    .collect::<Vec<u32>>();
                if !idx_data_map.contains_key(&idx) {
                    break;
                }
            }

            let data: Vec<u32> = (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect();
            idx_data_map.insert(idx.clone(), data.clone());
            btree.update(&idx, &data);
            ops.push(Operation::new(clk, idx, data, OpType::Write));
        } else {
            let idx = idx_data_map
                .iter()
                .nth(rng.gen::<usize>() % idx_data_map.len())
                .unwrap()
                .0
                .to_vec();

            let op_type = {
                if rng.gen::<bool>() {
                    OpType::Read
                } else {
                    OpType::Write
                }
            };

            let data = {
                if op_type == OpType::Read {
                    idx_data_map[&idx].to_vec()
                } else {
                    (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect()
                }
            };

            if op_type == OpType::Write {
                idx_data_map.insert(idx.clone(), data.clone());
                btree.update(&idx, &data);
            }

            ops.push(Operation::new(clk, idx, data, op_type));
        }
    }
    let final_pages = btree.gen_all_trace(committer);
    (init_pages, false, final_pages, false, ops)
}

// generates a test case where one leaf is not exposed
fn generate_mixed_ops_remove_first_leaf(
    idx_len: usize,
    data_len: usize,
    page_height: usize,
    limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>) {
    let (mut btree, clks, mut init_pages, mut idx_data_map) = generate_init_tree_and_clks(
        idx_len,
        data_len,
        page_height,
        limb_bits,
        num_ops,
        committer,
    );
    let mut rng = create_seeded_rng();
    let mut ops: Vec<Operation> = vec![];
    for clk in clks {
        if rng.gen::<bool>() {
            let mut idx;
            loop {
                idx = (0..idx_len)
                    .map(|_| rng.gen::<u32>() % MAX_LIMB_VAL)
                    .collect::<Vec<u32>>();
                if !idx_data_map.contains_key(&idx) && idx[0] > MAX_LIMB_VAL / 2 {
                    break;
                }
            }

            let data: Vec<u32> = (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect();
            idx_data_map.insert(idx.clone(), data.clone());
            btree.update(&idx, &data);
            ops.push(Operation::new(clk, idx, data, OpType::Write));
        } else {
            let mut idx;
            loop {
                idx = idx_data_map
                    .iter()
                    .nth(rng.gen::<usize>() % idx_data_map.len())
                    .unwrap()
                    .0
                    .to_vec();
                if idx[0] > MAX_LIMB_VAL / 2 {
                    break;
                }
            }

            let op_type = {
                if rng.gen::<bool>() {
                    OpType::Read
                } else {
                    OpType::Write
                }
            };

            let data = {
                if op_type == OpType::Read {
                    idx_data_map[&idx].to_vec()
                } else {
                    (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect()
                }
            };

            if op_type == OpType::Write {
                idx_data_map.insert(idx.clone(), data.clone());
                btree.update(&idx, &data);
            }

            ops.push(Operation::new(clk, idx, data, op_type));
        }
    }
    let mut final_pages = btree.gen_all_trace(committer);
    final_pages.leaf_pages.pop();
    init_pages.leaf_pages.pop();
    (init_pages.clone(), false, final_pages.clone(), false, ops)
}

// generates a test case where we do all kinds of ops to an empty tree
fn generate_mixed_ops_empty_start(
    idx_len: usize,
    data_len: usize,
    page_height: usize,
    limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>) {
    let mut rng = create_seeded_rng();
    let mut btree = PageBTree::<BABYBEAR_COMMITMENT_LEN>::new(
        limb_bits,
        idx_len,
        data_len,
        page_height,
        page_height,
    );
    let mut idx_data_map = HashMap::new();
    let init_pages = btree.gen_all_trace(committer);

    let mut clks: Vec<usize> = (0..num_ops)
        .map(|_| rng.gen::<usize>() % (MAX_VAL as usize))
        .collect();
    clks.sort();
    let mut ops: Vec<Operation> = vec![];
    for (i, clk) in clks.into_iter().enumerate() {
        if rng.gen::<bool>() || i == 0 {
            let mut idx;
            loop {
                idx = (0..idx_len)
                    .map(|_| rng.gen::<u32>() % MAX_LIMB_VAL)
                    .collect::<Vec<u32>>();
                if !idx_data_map.contains_key(&idx) {
                    break;
                }
            }

            let data: Vec<u32> = (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect();
            idx_data_map.insert(idx.clone(), data.clone());
            btree.update(&idx, &data);
            ops.push(Operation::new(clk, idx, data, OpType::Write));
        } else {
            let idx = idx_data_map
                .iter()
                .nth(rng.gen::<usize>() % idx_data_map.len())
                .unwrap()
                .0
                .to_vec();

            let op_type = {
                if rng.gen::<bool>() {
                    OpType::Read
                } else {
                    OpType::Write
                }
            };

            let data = {
                if op_type == OpType::Read {
                    idx_data_map[&idx].to_vec()
                } else {
                    (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect()
                }
            };

            if op_type == OpType::Write {
                idx_data_map.insert(idx.clone(), data.clone());
                btree.update(&idx, &data);
            }

            ops.push(Operation::new(clk, idx, data, op_type));
        }
    }
    let final_pages = btree.gen_all_trace(committer);
    (init_pages, true, final_pages, true, ops)
}

// generates a test case where we add no new keys to a large tree
fn generate_large_tree_no_new_keys(
    _idx_len: usize,
    data_len: usize,
    _page_height: usize,
    _limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>) {
    let mut rng = create_seeded_rng();
    let mut btree = PageBTree::<BABYBEAR_COMMITMENT_LEN>::load(vec![
        639955356, 1577306122, 107201956, 1528176068, 704402408, 1775238984, 169542638, 1916258191,
    ])
    .unwrap();
    let existing_keys = vec![vec![534524, 887809], vec![380587, 701877]];
    for k in &existing_keys {
        btree.search(k).unwrap();
    }
    let init_pages = btree.gen_all_trace(committer);
    btree.consistency_check();
    let mut clks: Vec<usize> = (0..num_ops)
        .map(|_| rng.gen::<usize>() % (MAX_VAL as usize))
        .collect();
    clks.sort();
    let mut ops: Vec<Operation> = vec![];
    for clk in clks {
        let idx = existing_keys[rng.gen::<usize>() % existing_keys.len()].clone();

        let op_type = {
            if rng.gen::<bool>() {
                OpType::Read
            } else {
                OpType::Write
            }
        };

        let data = {
            if op_type == OpType::Read {
                btree.search(&idx).unwrap()
            } else {
                (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect()
            }
        };

        if op_type == OpType::Write {
            btree.update(&idx, &data);
        }

        ops.push(Operation::new(clk, idx, data, op_type));
    }
    let final_pages = btree.gen_all_trace(committer);
    (init_pages.clone(), false, final_pages.clone(), false, ops)
}

// generates a test case where we do all kinds of ops to a large tree
fn generate_large_tree_new_keys(
    idx_len: usize,
    data_len: usize,
    _page_height: usize,
    _limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>) {
    let mut rng = create_seeded_rng();
    let mut btree = PageBTree::<BABYBEAR_COMMITMENT_LEN>::load(vec![
        639955356, 1577306122, 107201956, 1528176068, 704402408, 1775238984, 169542638, 1916258191,
    ])
    .unwrap();
    let existing_keys = [vec![534524, 887809], vec![380587, 701877]];
    let mut clks: Vec<usize> = (0..num_ops)
        .map(|_| rng.gen::<usize>() % (MAX_VAL as usize))
        .collect();
    clks.sort();
    let mut ops: Vec<Operation> = vec![];
    for (i, clk) in clks.into_iter().enumerate() {
        if rng.gen::<bool>() || i == 0 {
            let idx = (0..idx_len)
                .map(|_| rng.gen::<u32>() % 1000 + 1000000)
                .collect::<Vec<u32>>();

            let data: Vec<u32> = (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect();
            btree.update(&idx, &data);
            ops.push(Operation::new(clk, idx, data, OpType::Write));
        } else {
            let idx = existing_keys[rng.gen::<usize>() % existing_keys.len()].clone();

            let op_type = {
                if rng.gen::<bool>() {
                    OpType::Read
                } else {
                    OpType::Write
                }
            };

            let data = {
                if op_type == OpType::Read {
                    btree.search(&idx).unwrap()
                } else {
                    (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect()
                }
            };

            if op_type == OpType::Write {
                btree.update(&idx, &data);
            }

            ops.push(Operation::new(clk, idx, data, op_type));
        }
    }
    let final_pages = btree.gen_all_trace(committer);
    let init_pages = btree.gen_loaded_trace();
    (init_pages.clone(), false, final_pages.clone(), false, ops)
}
