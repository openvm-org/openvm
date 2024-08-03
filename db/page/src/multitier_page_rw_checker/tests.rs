use std::{collections::HashMap, iter, sync::Arc};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::{
    keygen::{types::MultiStarkProvingKey, MultiStarkKeygenBuilder},
    prover::{
        trace::{TraceCommitmentBuilder, TraceCommitter},
        MultiTraceStarkProver,
    },
    verifier::VerificationError,
};
use afs_test_utils::{
    config::{
        self,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
    },
    interaction::dummy_interaction_air::DummyInteractionAir,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

use super::page_controller;
use crate::{
    multitier_page_rw_checker::page_controller::{
        MyLessThanTupleParams, PageController, PageTreeParams,
    },
    page_btree::{PageBTree, PageBTreePages},
    page_rw_checker::page_controller::{OpType, Operation},
};

pub const BABYBEAR_COMMITMENT_LEN: usize = 8;
pub const DECOMP_BITS: usize = 6;
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
    let data_bus_index = 0;
    let internal_data_bus_index = 1;
    let lt_bus_index = 2;
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

    let init_path_bus = 3;
    let final_path_bus = 4;
    let ops_bus_index = 5;

    let init_param = PageTreeParams {
        path_bus_index: init_path_bus,
        leaf_cap: 8,
        internal_cap: 24,
        leaf_page_height: page_height,
        internal_page_height: page_height,
    };

    let final_param = PageTreeParams {
        path_bus_index: final_path_bus,
        leaf_cap: 8,
        internal_cap: 24,
        leaf_page_height: page_height,
        internal_page_height: page_height,
    };

    let less_than_tuple_param = MyLessThanTupleParams {
        limb_bits: 20,
        decomp: DECOMP_BITS,
    };

    let range_checker = Arc::new(RangeCheckerGateChip::new(lt_bus_index, 1 << DECOMP_BITS));

    let mut page_controller: PageController<BabyBearPoseidon2Config, BABYBEAR_COMMITMENT_LEN> =
        PageController::new(
            data_bus_index,
            internal_data_bus_index,
            ops_bus_index,
            lt_bus_index,
            idx_len,
            data_len,
            init_param.clone(),
            final_param.clone(),
            less_than_tuple_param,
            range_checker,
        );
    let ops_sender = DummyInteractionAir::new(idx_len + data_len + 2, true, ops_bus_index);
    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    page_controller.set_up_keygen_builder(&mut keygen_builder, &ops_sender);

    let pk = keygen_builder.generate_pk();
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
        init_pages.leaf_pages,
        init_pages.internal_pages,
        init_root_is_leaf,
        0,
        final_pages.leaf_pages,
        final_pages.internal_pages,
        final_root_is_leaf,
        0,
        &ops,
        num_ops,
        &ops_sender,
        &mut page_controller,
        &mut trace_builder,
        &pk,
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
    page_controller: &mut page_controller::PageController<
        BabyBearPoseidon2Config,
        BABYBEAR_COMMITMENT_LEN,
    >,
    trace_builder: &mut TraceCommitmentBuilder<BabyBearPoseidon2Config>,
    pk: &MultiStarkProvingKey<BabyBearPoseidon2Config>,
    trace_degree: usize,
) -> Result<(), VerificationError> {
    page_controller.range_checker.clear();
    let prover_data = page_controller.load_page_and_ops(
        init_leaf_pages,
        init_internal_pages,
        init_root_is_leaf,
        init_root_idx,
        final_leaf_pages,
        final_internal_pages,
        final_root_is_leaf,
        final_root_idx,
        ops,
        trace_degree,
        &mut trace_builder.committer,
        None,
        None,
    );
    let ops_sender_trace = RowMajorMatrix::new(
        ops.iter()
            .flat_map(|op| {
                iter::once(Val::one())
                    .chain(iter::once(Val::from_canonical_usize(op.clk)))
                    .chain(iter::once(Val::from_canonical_u8(op.op_type as u8)))
                    .chain(op.idx.iter().map(|x| Val::from_canonical_u32(*x)))
                    .chain(op.data.iter().map(|x| Val::from_canonical_u32(*x)))
            })
            .chain(
                iter::repeat_with(|| iter::repeat(Val::zero()).take(1 + ops_sender.field_width()))
                    .take(num_ops - ops.len())
                    .flatten(),
            )
            .collect(),
        1 + ops_sender.field_width(),
    );
    let (proof, pis) = page_controller.prove(
        engine,
        pk,
        trace_builder,
        prover_data,
        ops_sender,
        ops_sender_trace,
    );
    page_controller.verify(engine, &pk.vk(), &proof, &pis, ops_sender)
}

fn generate_no_new_keys(
    idx_len: usize,
    data_len: usize,
    page_height: usize,
    limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>) {
    const MAX_LIMB_VAL: u32 = 1 << 20;
    let mut rng = create_seeded_rng();
    let mut btree = PageBTree::<BABYBEAR_COMMITMENT_LEN>::new(
        limb_bits,
        idx_len,
        data_len,
        page_height,
        page_height,
        "".to_owned(),
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
    let init_pages = btree.gen_all_trace(committer, None);
    // check for correctness
    for (idx, _) in idx_data_map.iter() {
        btree.search(idx).unwrap();
    }
    let mut clks: Vec<usize> = (0..num_ops)
        .map(|_| rng.gen::<usize>() % (MAX_VAL as usize))
        .collect();
    clks.sort();

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
    let final_pages = btree.gen_all_trace(committer, None);
    (init_pages, false, final_pages, false, ops)
}

fn generate_new_keys(
    idx_len: usize,
    data_len: usize,
    page_height: usize,
    limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>) {
    const MAX_LIMB_VAL: u32 = 1 << 20;
    let mut rng = create_seeded_rng();
    let mut btree = PageBTree::<BABYBEAR_COMMITMENT_LEN>::new(
        limb_bits,
        idx_len,
        data_len,
        page_height,
        page_height,
        "".to_owned(),
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
    let init_pages = btree.gen_all_trace(committer, None);
    // check for correctness
    for (idx, _) in idx_data_map.iter() {
        btree.search(idx).unwrap();
    }
    let mut clks: Vec<usize> = (0..num_ops)
        .map(|_| rng.gen::<usize>() % (MAX_VAL as usize))
        .collect();
    clks.sort();
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
    let final_pages = btree.gen_all_trace(committer, None);
    (init_pages, false, final_pages, false, ops)
}

fn generate_mixed_ops(
    idx_len: usize,
    data_len: usize,
    page_height: usize,
    limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>) {
    const MAX_LIMB_VAL: u32 = 1 << 20;
    let mut rng = create_seeded_rng();
    let mut btree = PageBTree::<BABYBEAR_COMMITMENT_LEN>::new(
        limb_bits,
        idx_len,
        data_len,
        page_height,
        page_height,
        "".to_owned(),
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
    let init_pages = btree.gen_all_trace(committer, None);
    // check for correctness
    for (idx, _) in idx_data_map.iter() {
        btree.search(idx).unwrap();
    }
    let mut clks: Vec<usize> = (0..num_ops)
        .map(|_| rng.gen::<usize>() % (MAX_VAL as usize))
        .collect();
    clks.sort();
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
    let final_pages = btree.gen_all_trace(committer, None);
    (init_pages, false, final_pages, false, ops)
}

fn generate_mixed_ops_remove_first_leaf(
    idx_len: usize,
    data_len: usize,
    page_height: usize,
    limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>) {
    const MAX_LIMB_VAL: u32 = 1 << 20;
    let mut rng = create_seeded_rng();
    let mut btree = PageBTree::<BABYBEAR_COMMITMENT_LEN>::new(
        limb_bits,
        idx_len,
        data_len,
        page_height,
        page_height,
        "".to_owned(),
    );
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
    let mut init_pages = btree.gen_all_trace(committer, None);
    // check for correctness
    for (idx, _) in idx_data_map.iter() {
        btree.search(idx).unwrap();
    }
    let mut clks: Vec<usize> = (0..num_ops)
        .map(|_| rng.gen::<usize>() % (MAX_VAL as usize))
        .collect();
    clks.sort();
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
    let mut final_pages = btree.gen_all_trace(committer, None);
    final_pages.leaf_pages.pop();
    init_pages.leaf_pages.pop();
    (init_pages.clone(), false, final_pages.clone(), false, ops)
}

fn generate_mixed_ops_empty_start(
    idx_len: usize,
    data_len: usize,
    page_height: usize,
    limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>) {
    const MAX_LIMB_VAL: u32 = 1 << 20;
    let mut rng = create_seeded_rng();
    let mut btree = PageBTree::<BABYBEAR_COMMITMENT_LEN>::new(
        limb_bits,
        idx_len,
        data_len,
        page_height,
        page_height,
        "".to_owned(),
    );
    let mut idx_data_map = HashMap::new();
    let init_pages = btree.gen_all_trace(committer, None);

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
    let final_pages = btree.gen_all_trace(committer, None);
    (init_pages, true, final_pages, true, ops)
}

fn generate_large_tree_no_new_keys(
    _idx_len: usize,
    data_len: usize,
    _page_height: usize,
    _limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>) {
    let mut rng = create_seeded_rng();
    let mut btree = PageBTree::<BABYBEAR_COMMITMENT_LEN>::load(
        "src/pagebtree".to_owned(),
        "large".to_owned(),
        "".to_owned(),
    )
    .unwrap();
    let existing_keys = vec![vec![534524, 887809], vec![380587, 701877]];
    for k in &existing_keys {
        btree.search(k).unwrap();
    }
    let init_pages = btree.gen_all_trace(committer, None);
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
    let final_pages = btree.gen_all_trace(committer, None);
    (init_pages.clone(), false, final_pages.clone(), false, ops)
}

fn generate_large_tree_new_keys(
    idx_len: usize,
    data_len: usize,
    _page_height: usize,
    _limb_bits: usize,
    num_ops: usize,
    committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> (PageBTreePages, bool, PageBTreePages, bool, Vec<Operation>) {
    let mut rng = create_seeded_rng();
    let mut btree = PageBTree::<BABYBEAR_COMMITMENT_LEN>::load(
        "src/pagebtree".to_owned(),
        "large".to_owned(),
        "".to_owned(),
    )
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
    let final_pages = btree.gen_all_trace(committer, None);
    let init_pages = btree.gen_loaded_trace();
    (init_pages.clone(), false, final_pages.clone(), false, ops)
}
