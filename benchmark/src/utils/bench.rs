use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use std::collections::HashSet;
use std::iter;

use rand::Rng;

use afs_page::common::page::Page;
use afs_page::page_rw_checker::page_controller::{OpType, Operation};
use afs_stark_backend::prover::trace::ProverTraceData;
use afs_stark_backend::prover::trace::TraceCommitter;
use afs_test_utils::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config,
    interaction::dummy_interaction_air::DummyInteractionAir, utils::create_seeded_rng,
};

type Val = BabyBear;

pub fn generate_page_and_ops(
    idx_len: usize,
    data_len: usize,
    page_height: usize,
    num_ops: usize,
    max_idx: u32,
    max_data: u32,
) -> (Page, Vec<Operation>) {
    let mut rng = create_seeded_rng();

    // Generating a random page with distinct indices
    let initial_page = Page::random(
        &mut rng,
        idx_len,
        data_len,
        max_idx,
        max_data,
        page_height,
        page_height,
    );

    // We will generate the final page from the initial page below
    // while generating the operations
    let mut final_page = initial_page.clone();

    // Generating random sorted distinct timestamps for operations
    let mut clks = HashSet::new();
    while clks.len() < num_ops {
        clks.insert(rng.gen::<usize>() % (max_data as usize - 2) + 1);
    }
    let mut clks: Vec<usize> = clks.into_iter().collect();
    clks.sort();

    let mut ops: Vec<Operation> = vec![];
    for &clk in clks.iter() {
        let op_type = {
            if rng.gen::<u32>() % 3 == 0 {
                OpType::Read
            } else if rng.gen::<u32>() % 3 == 1 {
                OpType::Write
            } else {
                OpType::Delete
            }
        };

        let mut idx = final_page.get_random_idx(&mut rng);

        // if this is a write operation, make it an insert sometimes
        if op_type == OpType::Write && rng.gen::<u32>() % 2 == 0 && !final_page.is_full() {
            idx = (0..idx_len).map(|_| rng.gen::<u32>() % max_idx).collect();
        }

        let data = {
            if op_type == OpType::Read {
                final_page[&idx].clone()
            } else if op_type == OpType::Write {
                (0..data_len).map(|_| rng.gen::<u32>() % max_data).collect()
            } else {
                vec![0; data_len]
            }
        };

        if op_type == OpType::Write {
            if final_page.contains(&idx) {
                final_page[&idx].clone_from(&data);
            } else {
                final_page.insert(&idx, &data);
            }
        } else if op_type == OpType::Delete {
            final_page.delete(&idx);
        }

        ops.push(Operation::new(clk, idx, data, op_type));
    }

    (initial_page, ops)
}

pub fn gen_ops_sender_trace(
    ops_sender: &DummyInteractionAir,
    ops: &[Operation],
) -> RowMajorMatrix<Val> {
    RowMajorMatrix::new(
        ops.iter()
            .flat_map(|op| {
                iter::once(Val::one())
                    .chain(iter::once(Val::from_canonical_usize(op.clk)))
                    .chain(iter::once(Val::from_canonical_u8(op.op_type as u8)))
                    .chain(op.idx.iter().map(|x| Val::from_canonical_u32(*x)))
                    .chain(op.data.iter().map(|x| Val::from_canonical_u32(*x)))
            })
            .collect(),
        1 + ops_sender.field_width(),
    )
}

pub fn get_dummy_ptd(
    trace_committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> ProverTraceData<BabyBearPoseidon2Config> {
    let simple_trace = RowMajorMatrix::new_col(vec![Val::from_canonical_u32(1)]);
    trace_committer.commit(vec![simple_trace])
}
