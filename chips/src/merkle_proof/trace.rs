use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::PseudoCompressionFunction;

use crate::{merkle_proof::MerkleProofOp, utils::FieldFrom};

use super::{
    columns::{num_merkle_proof_cols, MerkleProofCols},
    MerkleProofAir,
};

impl<const DEPTH: usize, const DIGEST_WIDTH: usize> MerkleProofAir<DEPTH, DIGEST_WIDTH> {
    pub fn generate_trace<F, T, Compress>(
        &self,
        operations: Vec<MerkleProofOp<T, DEPTH, DIGEST_WIDTH>>,
        hasher: &Compress,
    ) -> RowMajorMatrix<F>
    where
        F: PrimeField32 + FieldFrom<T>,
        T: Default + Copy,
        Compress: PseudoCompressionFunction<[T; DIGEST_WIDTH], 2>,
    {
        let num_merkle_proof_cols = num_merkle_proof_cols::<DEPTH, DIGEST_WIDTH>();

        let num_real_rows = operations.len() * DEPTH;
        let num_rows = num_real_rows.next_power_of_two();
        let mut trace = RowMajorMatrix::new(
            vec![F::zero(); num_rows * num_merkle_proof_cols],
            num_merkle_proof_cols,
        );
        let (prefix, rows, suffix) = unsafe {
            trace
                .values
                .align_to_mut::<MerkleProofCols<F, DEPTH, DIGEST_WIDTH>>()
        };
        assert!(prefix.is_empty(), "Alignment should match");
        assert!(suffix.is_empty(), "Alignment should match");
        assert_eq!(rows.len(), num_rows);

        for (leaf_rows, op) in rows.chunks_mut(DEPTH).zip(operations.iter()) {
            generate_trace_rows_for_op(leaf_rows, op, hasher);

            for row in leaf_rows.iter_mut() {
                row.is_real = F::one();
            }
        }

        // Fill padding rows
        for input_rows in rows.chunks_mut(DEPTH).skip(num_real_rows) {
            let op = MerkleProofOp::default();
            generate_trace_rows_for_op(input_rows, &op, hasher);
        }

        trace
    }
}

pub fn generate_trace_rows_for_op<F, T, Compress, const DEPTH: usize, const DIGEST_WIDTH: usize>(
    rows: &mut [MerkleProofCols<F, DEPTH, DIGEST_WIDTH>],
    op: &MerkleProofOp<T, DEPTH, DIGEST_WIDTH>,
    hasher: &Compress,
) where
    F: PrimeField32 + FieldFrom<T>,
    T: Default + Copy,
    Compress: PseudoCompressionFunction<[T; DIGEST_WIDTH], 2>,
{
    let MerkleProofOp {
        leaf_index,
        leaf_hash,
        siblings,
    } = op;

    // Fill the first row with the leaf.
    for (node_byte, &leaf_hash_byte) in rows[0].node.iter_mut().zip(leaf_hash.iter()) {
        *node_byte = F::from_val(leaf_hash_byte);
    }

    let mut node = generate_trace_row_for_round(
        &mut rows[0],
        0,
        leaf_index & 1,
        leaf_index & 1,
        leaf_hash,
        &siblings[0],
        hasher,
    );

    for round in 1..rows.len() {
        // Copy previous row's output to next row's input.
        for i in 0..DIGEST_WIDTH {
            rows[round].node[i] = rows[round - 1].output[i];
        }

        let mask = (1 << (round + 1)) - 1;
        node = generate_trace_row_for_round(
            &mut rows[round],
            round,
            leaf_index & mask,
            (leaf_index >> round) & 1,
            &node,
            &siblings[round],
            hasher,
        );
    }
}

pub fn generate_trace_row_for_round<F, T, Compress, const DEPTH: usize, const DIGEST_WIDTH: usize>(
    row: &mut MerkleProofCols<F, DEPTH, DIGEST_WIDTH>,
    round: usize,
    accumulate_index: usize,
    is_right_child: usize,
    node: &[T; DIGEST_WIDTH],
    sibling: &[T; DIGEST_WIDTH],
    hasher: &Compress,
) -> [T; DIGEST_WIDTH]
where
    F: PrimeField32 + FieldFrom<T>,
    T: Default + Copy,
    Compress: PseudoCompressionFunction<[T; DIGEST_WIDTH], 2>,
{
    row.step_flags[round] = F::one();

    let (left_node, right_node) = if is_right_child == 0 {
        (node, sibling)
    } else {
        (sibling, node)
    };

    let output = hasher.compress([*left_node, *right_node]);

    row.is_right_child = F::from_canonical_usize(is_right_child);
    row.accumulated_index = F::from_canonical_usize(accumulate_index);
    for i in 0..DIGEST_WIDTH {
        row.sibling[i] = F::from_val(sibling[i]);

        row.left_node[i] = F::from_val(left_node[i]);
        row.right_node[i] = F::from_val(right_node[i]);

        row.output[i] = F::from_val(output[i]);
    }

    output
}
