use std::borrow::Borrow;

use afs_primitives::utils::not;
use afs_stark_backend::{air_builders::sub::SubAirBuilder, interaction::InteractionBuilder};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_keccak_air::{KeccakAir, NUM_KECCAK_COLS, NUM_ROUNDS};
use p3_matrix::Matrix;

use super::columns::{KeccakPermuteCols, NUM_KECCAK_PERMUTE_COLS};

#[derive(Clone, Copy, Debug)]
pub struct KeccakPermuteAir {
    // TODO: add direct non-memory interactions
}

impl<F> BaseAir<F> for KeccakPermuteAir {
    fn width(&self) -> usize {
        NUM_KECCAK_PERMUTE_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for KeccakPermuteAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let [local, next] = [0, 1].map(|i| main.row_slice(i));
        let local: &KeccakPermuteCols<AB::Var> = (*local).borrow();
        let next: &KeccakPermuteCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.io.is_opcode);
        // All rounds of a single permutation must have same is_opcode, clk, dst, e (src, a, c are only read on the 0-th round right now)
        let mut transition_builder = builder.when_transition();
        let mut round_builder =
            transition_builder.when(not::<AB>(local.inner.step_flags[NUM_ROUNDS - 1]));
        round_builder.assert_eq(local.io.is_opcode, next.io.is_opcode);
        round_builder.assert_eq(local.io.clk, next.io.clk);
        round_builder.assert_eq(local.io.e, next.io.e);
        round_builder.assert_eq(local.aux.dst, next.aux.dst);

        // TODO: `d` should not be 0, this should be handled by memory chip directly
        // TODO: `e` should not be 0

        let keccak_air = KeccakAir {};
        let mut sub_builder =
            SubAirBuilder::<AB, KeccakAir, AB::Var>::new(builder, 0..NUM_KECCAK_COLS);
        keccak_air.eval(&mut sub_builder);

        self.eval_interactions(builder, local);
    }
}
