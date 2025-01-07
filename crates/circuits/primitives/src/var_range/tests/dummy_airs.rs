use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use crate::var_range::bus::VariableRangeCheckerBus;

// dummy AIR for testing VariableRangeCheckerBus::send
pub struct TestSendAir {
    bus: VariableRangeCheckerBus,
}

impl TestSendAir {
    pub fn new(bus: VariableRangeCheckerBus) -> Self {
        Self { bus }
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for TestSendAir {}
impl<F: Field> PartitionedBaseAir<F> for TestSendAir {}
impl<F: Field> BaseAir<F> for TestSendAir {
    fn width(&self) -> usize {
        2
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }
}

impl<AB: InteractionBuilder + AirBuilder> Air<AB> for TestSendAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        // local = [value, max_bits]
        let local = main.row_slice(0);
        self.bus.send(local[0], local[1]).eval(builder, AB::F::ONE);
    }
}

// dummy AIR for testing VariableRangeCheckerBus::range_check
pub struct TestRangeCheckAir {
    bus: VariableRangeCheckerBus,
    max_bits: usize,
}

impl TestRangeCheckAir {
    pub fn new(bus: VariableRangeCheckerBus, max_bits: usize) -> Self {
        Self { bus, max_bits }
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for TestRangeCheckAir {}
impl<F: Field> PartitionedBaseAir<F> for TestRangeCheckAir {}
impl<F: Field> BaseAir<F> for TestRangeCheckAir {
    fn width(&self) -> usize {
        1
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }
}

impl<AB: InteractionBuilder + AirBuilder> Air<AB> for TestRangeCheckAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        // local = [value]
        let local = main.row_slice(0);
        self.bus
            .range_check(local[0], self.max_bits)
            .eval(builder, AB::F::ONE);
    }
}
