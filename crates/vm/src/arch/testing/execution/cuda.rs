use std::{mem::size_of, slice::from_raw_parts};

use openvm_circuit::{
    arch::{
        testing::{execution::air::DummyExecutionInteractionCols, ExecutionTester},
        ExecutionBus, ExecutionState,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_cuda_backend::{base::DeviceMatrix, prover_backend::GpuBackend, types::F};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{prover::AirProvingContext, Chip};

use crate::cuda_abi::execution_testing;

pub struct DeviceExecutionTester(pub(crate) ExecutionTester<F>);

impl DeviceExecutionTester {
    pub fn new(bus: ExecutionBus) -> Self {
        Self(ExecutionTester::new(bus))
    }

    pub fn bus(&self) -> ExecutionBus {
        self.0.bus
    }

    pub fn execute(
        &mut self,
        initial_state: ExecutionState<u32>,
        final_state: ExecutionState<u32>,
    ) {
        self.0.execute(initial_state, final_state);
    }
}

impl<RA> Chip<RA, GpuBackend> for DeviceExecutionTester {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let height = next_power_of_two_or_zero(self.0.records.len());
        let width = size_of::<DummyExecutionInteractionCols<F>>();

        if height == 0 {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        let trace = DeviceMatrix::<F>::with_capacity(height, width);

        let records = &self.0.records;
        let num_records = records.len();

        unsafe {
            let bytes_size = num_records * size_of::<DummyExecutionInteractionCols<F>>();
            let records_bytes = from_raw_parts(records.as_ptr() as *const u8, bytes_size);
            let records = records_bytes.to_device().unwrap();
            execution_testing::tracegen(trace.buffer(), height, width, &records).unwrap();
        }
        AirProvingContext::simple_no_pis(trace)
    }
}
