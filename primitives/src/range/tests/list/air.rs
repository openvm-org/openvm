use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::{ListCols, NUM_LIST_COLS};

#[derive(Copy, Clone, Debug)]
pub struct ListAir {
    /// The index for the Range Checker bus.
    pub bus_index: usize,
}

impl<F: Field> BaseAir<F> for ListAir {
    fn width(&self) -> usize {
        NUM_LIST_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for ListAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ListCols<AB::Var> = (*local).borrow();

        // We do not implement SubAirBridge trait for brevity
        builder.push_send(self.bus_index, vec![local.val], AB::F::one());
    }
}
