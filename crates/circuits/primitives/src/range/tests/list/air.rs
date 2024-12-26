use std::borrow::Borrow;

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::{Field, FieldAlgebra},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use super::columns::{ListCols, NUM_LIST_COLS};
use crate::range::bus::RangeCheckBus;

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct ListAir {
    /// The index for the Range Checker bus.
    pub bus: RangeCheckBus,
}

impl<F: Field> BaseAirWithPublicValues<F> for ListAir {}
impl<F: Field> PartitionedBaseAir<F> for ListAir {}
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
        self.bus.send(local.val).eval(builder, AB::F::ONE);
    }
}
