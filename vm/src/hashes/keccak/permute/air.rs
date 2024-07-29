use std::borrow::Borrow;

use afs_stark_backend::{air_builders::sub::SubAirBuilder, interaction::InteractionBuilder};
use p3_air::{Air, BaseAir};
use p3_keccak_air::{KeccakAir, NUM_KECCAK_COLS};
use p3_matrix::Matrix;

use super::columns::{KeccakPermuteCols, NUM_KECCAK_PERMUTE_COLS};

#[derive(Clone, Copy, Debug)]
pub struct KeccakPermuteAir {
    pub input_bus: usize,
    pub output_bus: usize,
}

impl<F> BaseAir<F> for KeccakPermuteAir {
    fn width(&self) -> usize {
        NUM_KECCAK_PERMUTE_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for KeccakPermuteAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &KeccakPermuteCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_direct);

        let keccak_air = KeccakAir {};
        let mut sub_builder =
            SubAirBuilder::<AB, KeccakAir, AB::Var>::new(builder, 0..NUM_KECCAK_COLS);
        keccak_air.eval(&mut sub_builder);

        self.eval_interactions(builder, local);
    }
}
