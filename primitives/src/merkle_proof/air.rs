use core::borrow::Borrow;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use super::{
    columns::{num_merkle_proof_cols, MerkleProofCols},
    round_flags::eval_round_flags,
    MerkleProofAir,
};

impl<F, const DEPTH: usize, const DIGEST_WIDTH: usize> PartitionedBaseAir<F>
    for MerkleProofAir<DEPTH, DIGEST_WIDTH>
{ }
impl<F, const DEPTH: usize, const DIGEST_WIDTH: usize> BaseAir<F>
    for MerkleProofAir<DEPTH, DIGEST_WIDTH>
{
    fn width(&self) -> usize {
        num_merkle_proof_cols::<DEPTH, DIGEST_WIDTH>()
    }
}

impl<AB: AirBuilder, const DEPTH: usize, const DIGEST_WIDTH: usize> Air<AB>
    for MerkleProofAir<DEPTH, DIGEST_WIDTH>
{
    fn eval(&self, builder: &mut AB) {
        eval_round_flags::<_, DEPTH, DIGEST_WIDTH>(builder);

        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &MerkleProofCols<AB::Var, DEPTH, DIGEST_WIDTH> = (*local).borrow();
        let next: &MerkleProofCols<AB::Var, DEPTH, DIGEST_WIDTH> = (*next).borrow();

        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_right_child);

        let is_first_step = local.step_flags[0];
        let is_final_step = local.step_flags[DEPTH - 1];

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
        for i in 0..DIGEST_WIDTH {
            let diff = local.node[i] - local.sibling[i];
            let left = local.node[i] - local.is_right_child * diff.clone();
            let right = local.sibling[i] + local.is_right_child * diff;

            builder.assert_eq(left, local.left_node[i]);
            builder.assert_eq(right, local.right_node[i]);
        }

        // Output is copied to the next row.
        for i in 0..DIGEST_WIDTH {
            builder
                .when_ne(is_final_step, AB::Expr::one())
                .assert_eq(local.output[i], next.node[i]);
        }
    }
}
