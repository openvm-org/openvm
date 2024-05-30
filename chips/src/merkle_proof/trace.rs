use p3_field::PrimeField32;
use p3_keccak::KeccakF;
use p3_keccak_air::U64_LIMBS;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::{PseudoCompressionFunction, TruncatedPermutation};

use super::{
    columns::{num_merkle_proof_cols, MerkleProofCols, NUM_U64_HASH_ELEMS},
    MerkleProofChip, NUM_U8_HASH_ELEMS,
};

impl<const DEPTH: usize> MerkleProofChip<DEPTH> {
    pub fn generate_trace<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let num_merkle_proof_cols = num_merkle_proof_cols::<DEPTH>();

        let num_real_rows = self.siblings.iter().map(|s| s.len()).sum::<usize>();
        let num_rows = num_real_rows.next_power_of_two();
        let mut trace = RowMajorMatrix::new(
            vec![F::zero(); num_rows * num_merkle_proof_cols],
            num_merkle_proof_cols,
        );
        let (prefix, rows, suffix) =
            unsafe { trace.values.align_to_mut::<MerkleProofCols<F, DEPTH>>() };
        assert!(prefix.is_empty(), "Alignment should match");
        assert!(suffix.is_empty(), "Alignment should match");
        assert_eq!(rows.len(), num_rows);

        for (i, ((leaf, &leaf_index), siblings)) in self
            .leaves
            .iter()
            .zip(self.leaf_indices.iter())
            .zip(self.siblings.iter())
            .enumerate()
        {
            let leaf_rows = &mut rows[i * DEPTH..(i + 1) * DEPTH];
            generate_trace_rows_for_leaf(leaf_rows, leaf, leaf_index, siblings);

            for row in leaf_rows.iter_mut() {
                row.is_real = F::one();
            }
        }

        // Fill padding rows
        for input_rows in rows.chunks_mut(1).skip(num_real_rows) {
            generate_trace_rows_for_leaf(
                input_rows,
                &[0; NUM_U8_HASH_ELEMS],
                0,
                &[[0; NUM_U8_HASH_ELEMS]; DEPTH],
            );
        }

        trace
    }
}

pub fn generate_trace_rows_for_leaf<F: PrimeField32, const DEPTH: usize>(
    rows: &mut [MerkleProofCols<F, DEPTH>],
    leaf_hash: &[u8; NUM_U8_HASH_ELEMS],
    leaf_index: usize,
    siblings: &[[u8; NUM_U8_HASH_ELEMS]; DEPTH],
) {
    // Fill the first row with the leaf.
    for (x, input) in leaf_hash
        .chunks(NUM_U8_HASH_ELEMS / NUM_U64_HASH_ELEMS)
        .enumerate()
    {
        for limb in 0..U64_LIMBS {
            let limb_range = limb * 2..(limb + 1) * 2;
            rows[0].node[x][limb] =
                F::from_canonical_u16(u16::from_le_bytes(input[limb_range].try_into().unwrap()));
        }
    }

    let mut node = generate_trace_row_for_round(
        &mut rows[0],
        0,
        leaf_index & 1,
        leaf_index & 1,
        leaf_hash,
        &siblings[0],
    );

    for round in 1..rows.len() {
        // Copy previous row's output to next row's input.
        for x in 0..NUM_U64_HASH_ELEMS {
            for limb in 0..U64_LIMBS {
                rows[round].node[x][limb] = rows[round - 1].output[x][limb];
            }
        }

        let mask = (1 << (round + 1)) - 1;
        node = generate_trace_row_for_round(
            &mut rows[round],
            round,
            leaf_index & mask,
            (leaf_index >> round) & 1,
            &node,
            &siblings[round],
        );
    }
}

pub fn generate_trace_row_for_round<F: PrimeField32, const DEPTH: usize>(
    row: &mut MerkleProofCols<F, DEPTH>,
    round: usize,
    accumulate_index: usize,
    is_right_child: usize,
    node: &[u8; NUM_U8_HASH_ELEMS],
    sibling: &[u8; NUM_U8_HASH_ELEMS],
) -> [u8; NUM_U8_HASH_ELEMS] {
    row.step_flags[round] = F::one();

    let (left_node, right_node) = if is_right_child == 0 {
        (node, sibling)
    } else {
        (sibling, node)
    };

    let keccak = TruncatedPermutation::new(KeccakF {});
    let output = keccak.compress([*left_node, *right_node]);

    row.is_right_child = F::from_canonical_usize(is_right_child);
    row.accumulated_index = F::from_canonical_usize(accumulate_index);
    for x in 0..NUM_U64_HASH_ELEMS {
        let offset = x * NUM_U8_HASH_ELEMS / NUM_U64_HASH_ELEMS;
        for limb in 0..U64_LIMBS {
            let limb_range = (offset + limb * 2)..(offset + (limb + 1) * 2);

            row.sibling[x][limb] = F::from_canonical_u16(u16::from_le_bytes(
                sibling[limb_range.clone()].try_into().unwrap(),
            ));

            row.left_node[x][limb] = F::from_canonical_u16(u16::from_le_bytes(
                left_node[limb_range.clone()].try_into().unwrap(),
            ));
            row.right_node[x][limb] = F::from_canonical_u16(u16::from_le_bytes(
                right_node[limb_range.clone()].try_into().unwrap(),
            ));

            row.output[x][limb] =
                F::from_canonical_u16(u16::from_le_bytes(output[limb_range].try_into().unwrap()));
        }
    }

    output
}
