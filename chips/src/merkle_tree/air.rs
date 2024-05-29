use core::borrow::Borrow;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use super::{
    columns::{
        MerkleTreeCols, MERKLE_TREE_DEPTH, NUM_MERKLE_TREE_COLS, NUM_U16_LIMBS, NUM_U64_HASH_ELEMS,
    },
    round_flags::eval_round_flags,
    MerkleTreeChip,
};

impl<F> BaseAir<F> for MerkleTreeChip {
    fn width(&self) -> usize {
        NUM_MERKLE_TREE_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for MerkleTreeChip {
    fn eval(&self, builder: &mut AB) {
        eval_round_flags(builder);

        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &MerkleTreeCols<AB::Var> = (*local).borrow();
        let next: &MerkleTreeCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_right_child);

        let is_first_step = local.step_flags[0];
        let is_final_step = local.step_flags[MERKLE_TREE_DEPTH - 1];

        // Accumulated index is computed correctly
        builder
            .when(is_first_step)
            .assert_eq(local.accumulated_index, local.is_right_child);
        let bit_factor: AB::Expr = local
            .step_flags
            .iter()
            .enumerate()
            .map(|(i, &flag)| flag * AB::Expr::from_canonical_usize(1 << i))
            .sum();
        builder.when_ne(is_final_step, AB::Expr::one()).assert_eq(
            next.accumulated_index,
            local.accumulated_index + bit_factor * local.is_right_child,
        );

        // Left and right nodes are selected correctly.
        for i in 0..NUM_U64_HASH_ELEMS {
            for j in 0..NUM_U16_LIMBS {
                let diff = local.node[i][j] - local.sibling[i][j];
                let left = local.node[i][j] - local.is_right_child * diff.clone();
                let right = local.sibling[i][j] + local.is_right_child * diff;

                builder.assert_eq(left, local.left_node[i][j]);
                builder.assert_eq(right, local.right_node[i][j]);
            }
        }

        // Output is copied to the next row.
        for i in 0..NUM_U64_HASH_ELEMS {
            for j in 0..NUM_U16_LIMBS {
                builder
                    .when_ne(is_final_step, AB::Expr::one())
                    .assert_eq(local.output[i][j], next.node[i][j]);
            }
        }
    }
}
