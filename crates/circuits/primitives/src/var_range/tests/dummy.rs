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

#[cfg(feature = "cuda")]
pub mod cuda {
    use std::sync::Arc;

    use openvm_cuda_backend::{base::DeviceMatrix, prover_backend::GpuBackend, types::F};
    use openvm_cuda_common::{copy::MemCopyH2D as _, d_buffer::DeviceBuffer};
    use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

    use crate::{
        cuda_abi::var_range::dummy_tracegen,
        var_range::{VariableRangeCheckerChipGPU, NUM_VARIABLE_RANGE_PREPROCESSED_COLS},
    };

    pub struct DummyInteractionChipGPU {
        pub range_checker: Arc<VariableRangeCheckerChipGPU>,
        pub data: DeviceBuffer<u32>,
    }

    /// Expects trace to be: [1, value, bits]
    impl DummyInteractionChipGPU {
        pub fn new(range_checker: Arc<VariableRangeCheckerChipGPU>, data: Vec<u32>) -> Self {
            assert!(!data.is_empty());
            Self {
                range_checker,
                data: data.to_device().unwrap(),
            }
        }
    }

    impl<RA> Chip<RA, GpuBackend> for DummyInteractionChipGPU {
        fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
            let height = self.data.len();
            let width = NUM_VARIABLE_RANGE_PREPROCESSED_COLS + 1;
            let trace = DeviceMatrix::<F>::with_capacity(height, width);
            unsafe {
                dummy_tracegen(&self.data, trace.buffer(), &self.range_checker.count).unwrap();
            }
            AirProvingContext::simple_no_pis(trace)
        }
    }
}
