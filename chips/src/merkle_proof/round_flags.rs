//! Copied from p3/keccak-air under MIT license

use core::borrow::Borrow;

use p3_air::AirBuilder;
use p3_matrix::Matrix;

use super::columns::MerkleProofCols;

// TODO: Use sub-air builder
#[inline]
pub(crate) fn eval_round_flags<AB: AirBuilder, const DEPTH: usize, const DIGEST_WIDTH: usize>(
    builder: &mut AB,
) {
    let main = builder.main();
    let (local, next) = (main.row_slice(0), main.row_slice(1));
    let local: &MerkleProofCols<AB::Var, DEPTH, DIGEST_WIDTH> = (*local).borrow();
    let next: &MerkleProofCols<AB::Var, DEPTH, DIGEST_WIDTH> = (*next).borrow();

    // Initially, the first step flag should be 1 while the others should be 0.
    builder.when_first_row().assert_one(local.step_flags[0]);
    for i in 1..DEPTH {
        builder.when_first_row().assert_zero(local.step_flags[i]);
    }

    for i in 0..DEPTH {
        let current_round_flag = local.step_flags[i];
        let next_round_flag = next.step_flags[(i + 1) % DEPTH];
        builder
            .when_transition()
            .assert_eq(next_round_flag, current_round_flag);
    }
}
