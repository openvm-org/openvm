use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::byte_op_lookup::bus::ByteOperationLookupBus;

pub struct DummyAir {
    bus: ByteOperationLookupBus,
}

impl DummyAir {
    pub fn new(bus: ByteOperationLookupBus) -> Self {
        Self { bus }
    }
}

impl<F: Field> BaseAir<F> for DummyAir {
    fn width(&self) -> usize {
        4
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }
}

impl<AB: InteractionBuilder + AirBuilder> Air<AB> for DummyAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        // local = [x, y, z, op]
        let local = main.row_slice(0);
        self.bus
            .send(local[0], local[1], local[2], local[3])
            .eval(builder, AB::F::one());
    }
}
