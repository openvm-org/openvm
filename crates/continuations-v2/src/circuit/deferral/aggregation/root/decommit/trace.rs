use std::borrow::BorrowMut;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_stark_backend::prover::{AirProvingContext, ColMajorMatrix, CpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    circuit::{
        deferral::aggregation::root::decommit::air::MerkleDecommitCols,
        subair::{generate_cols_from_leaf_children, MerkleTreeCols},
    },
    utils::digests_to_poseidon2_input,
};

pub type IoCommit = ([F; DIGEST_SIZE], [F; DIGEST_SIZE]);

pub struct MerkleDecommitTraceCtx {
    pub proving_ctx: AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
    pub poseidon2_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub io_commits: Vec<IoCommit>,
    pub merkle_root: [F; DIGEST_SIZE],
}

pub fn generate_proving_ctx(
    leaf_children: Vec<IoCommit>,
    num_real_leaves: usize,
) -> MerkleDecommitTraceCtx {
    assert!(
        !leaf_children.is_empty(),
        "deferral root Merkle decommit requires at least one leaf"
    );
    assert!(
        leaf_children.len().is_power_of_two(),
        "deferral root Merkle decommit requires a power-of-two number of leaves"
    );
    assert!(
        (1..=leaf_children.len()).contains(&num_real_leaves),
        "deferral root Merkle decommit requires 1 <= num_real_leaves <= num_leaves"
    );
    let merkle_rows: Vec<MerkleTreeCols<F>> = generate_cols_from_leaf_children(leaf_children);
    let width = MerkleDecommitCols::<u8>::width();
    let height = merkle_rows.len();
    let num_rows_f = F::from_usize(height);
    let mut trace = vec![F::ZERO; height * width];
    let mut poseidon2_inputs = Vec::with_capacity(height.saturating_sub(1));
    let mut io_commits = Vec::with_capacity(num_real_leaves);

    for (row_idx, merkle_row) in merkle_rows.iter().copied().enumerate() {
        let cols: &mut MerkleDecommitCols<F> =
            trace[row_idx * width..(row_idx + 1) * width].borrow_mut();
        cols.merkle_tree_cols = merkle_row;
        cols.num_rows = num_rows_f;

        let is_leaf = merkle_row.receive_type == F::ONE;
        let should_send_commit = is_leaf && row_idx < num_real_leaves;
        cols.send_commits = F::from_bool(should_send_commit);

        if merkle_row.send_type != F::ZERO {
            poseidon2_inputs.push(digests_to_poseidon2_input(
                merkle_row.left_child,
                merkle_row.right_child,
            ));
        }
        if should_send_commit {
            io_commits.push((merkle_row.left_child, merkle_row.right_child));
        }
    }

    let merkle_root = merkle_rows[height - 2].parent;

    MerkleDecommitTraceCtx {
        proving_ctx: AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(
            &RowMajorMatrix::new(trace, width),
        )),
        poseidon2_inputs,
        io_commits,
        merkle_root,
    }
}
