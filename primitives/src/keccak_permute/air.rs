use afs_stark_backend::air_builders::sub::SubAirBuilder;
use core::borrow::Borrow;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_keccak_air::{KeccakAir, NUM_KECCAK_COLS, NUM_ROUNDS};
use p3_matrix::Matrix;

use super::{
    columns::{KeccakPermuteCols, NUM_KECCAK_PERMUTE_COLS},
    KeccakPermuteAir,
};

impl<F> BaseAir<F> for KeccakPermuteAir {
    fn width(&self) -> usize {
        NUM_KECCAK_PERMUTE_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for KeccakPermuteAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &KeccakPermuteCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);
        builder.assert_eq(
            local.is_real * local.keccak.step_flags[0],
            local.is_real_input,
        );
        builder.assert_eq(
            local.is_real * local.keccak.step_flags[NUM_ROUNDS - 1],
            local.is_real_output,
        );

        let keccak_air = KeccakAir {};
        let mut sub_builder =
            SubAirBuilder::<AB, KeccakAir, AB::Var>::new(builder, 0..NUM_KECCAK_COLS);
        keccak_air.eval(&mut sub_builder);
    }
}
