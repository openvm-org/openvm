use core::borrow::Borrow;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use super::{
    columns::{MerkleTreeCols, NUM_MERKLE_TREE_COLS, NUM_U16_LIMBS, NUM_U64_HASH_ELEMS},
    MerkleTreeChip,
};

impl<F> BaseAir<F> for MerkleTreeChip {
    fn width(&self) -> usize {
        NUM_MERKLE_TREE_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for MerkleTreeChip {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &MerkleTreeCols<AB::Var> = (*local).borrow();
        let next: &MerkleTreeCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_first_step);
        builder.assert_bool(local.is_final_step);
        builder.assert_bool(local.is_right_child);

        // TODO: is_first_step and is_final_step
        // TODO: Without bit_factor

        builder
            .when(local.is_first_step)
            .assert_eq(local.bit_factor, AB::Expr::one());
        builder
            .when_ne(local.is_final_step, AB::Expr::one())
            .assert_eq(AB::Expr::two() * local.bit_factor, next.bit_factor);

        // Accumulated index is computed correctly
        builder
            .when(local.is_first_step)
            .assert_eq(local.accumulated_index, local.is_right_child);
        builder
            .when_ne(local.is_final_step, AB::Expr::one())
            .assert_eq(
                next.accumulated_index,
                local.accumulated_index + local.bit_factor * local.is_right_child,
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
                    .when_ne(local.is_final_step, AB::Expr::one())
                    .assert_eq(local.output[i][j], next.node[i][j]);
            }
        }
    }
}
