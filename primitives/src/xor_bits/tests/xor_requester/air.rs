use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::{XorRequesterCols, NUM_XOR_REQUESTER_COLS};

#[derive(Copy, Clone, Debug)]
pub struct XorRequesterAir {
    pub bus_index: usize,
}

impl<F: Field> BaseAir<F> for XorRequesterAir {
    fn width(&self) -> usize {
        NUM_XOR_REQUESTER_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for XorRequesterAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &XorRequesterCols<AB::Var> = (*local).borrow();

        // We do not implement SubAirBridge trait for brevity
        builder.push_send(
            self.bus_index,
            vec![local.x, local.y, local.z],
            AB::F::one(),
        );
    }
}
