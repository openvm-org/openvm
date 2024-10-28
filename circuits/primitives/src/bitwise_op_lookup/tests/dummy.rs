use ax_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::bitwise_op_lookup::bus::BitwiseOperationLookupBus;

pub struct DummyAir {
    bus: BitwiseOperationLookupBus,
}

impl DummyAir {
    pub fn new(bus: BitwiseOperationLookupBus) -> Self {
        Self { bus }
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for DummyAir {}
impl<F: Field> PartitionedBaseAir<F> for DummyAir {}
impl<F: Field> BaseAir<F> for DummyAir {
    fn width(&self) -> usize {
        3
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }
}

impl<AB: InteractionBuilder + AirBuilder> Air<AB> for DummyAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        self.bus
            .send_xor(local[0], local[1], local[2])
            .eval(builder, AB::F::one());
    }
}
