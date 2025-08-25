use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

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
        4
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
            .push(local[0], local[1], local[2], local[3], true)
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
        bitwise_op_lookup::BitwiseOperationLookupChipGPU,
        cuda_abi::bitwise_op_lookup::dummy_tracegen,
    };

    const RECORD_WIDTH: usize = 3;
    const NUM_COLS: usize = 5;

    pub struct DummyInteractionChipGPU<const NUM_BITS: usize> {
        pub bitwise: Arc<BitwiseOperationLookupChipGPU<NUM_BITS>>,
        pub data: DeviceBuffer<u32>,
    }

    /// Expects trace to be: [1, x, y, z, op]
    impl<const NUM_BITS: usize> DummyInteractionChipGPU<NUM_BITS> {
        pub fn new(bitwise: Arc<BitwiseOperationLookupChipGPU<NUM_BITS>>, data: Vec<u32>) -> Self {
            assert!(!data.is_empty());
            Self {
                bitwise,
                data: data.to_device().unwrap(),
            }
        }
    }

    impl<RA, const NUM_BITS: usize> Chip<RA, GpuBackend> for DummyInteractionChipGPU<NUM_BITS> {
        fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
            let height = self.data.len() / RECORD_WIDTH;
            let trace = DeviceMatrix::<F>::with_capacity(height, NUM_COLS);
            unsafe {
                dummy_tracegen(
                    trace.buffer(),
                    &self.data,
                    &self.bitwise.count,
                    NUM_BITS as u32,
                )
                .unwrap();
            }
            AirProvingContext::simple_no_pis(trace)
        }
    }
}
