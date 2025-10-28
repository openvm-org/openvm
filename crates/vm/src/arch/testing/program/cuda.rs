use std::slice::from_raw_parts;

use openvm_circuit::{
    arch::{
        instructions::instruction::Instruction, testing::program::ProgramTester, ExecutionState,
    },
    system::program::{ProgramBus, ProgramExecutionCols},
    utils::next_power_of_two_or_zero,
};
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip, ChipUsageGetter};

use crate::cuda_abi::program_testing;

pub struct DeviceProgramTester(ProgramTester<F>);

impl DeviceProgramTester {
    pub fn new(bus: ProgramBus) -> Self {
        Self(ProgramTester::new(bus))
    }

    pub fn bus(&self) -> ProgramBus {
        self.0.bus
    }

    pub fn execute(&mut self, instruction: &Instruction<F>, initial_state: &ExecutionState<u32>) {
        self.0.execute(instruction, initial_state);
    }
}

impl ChipUsageGetter for DeviceProgramTester {
    fn air_name(&self) -> String {
        self.0.air_name()
    }

    fn current_trace_height(&self) -> usize {
        self.0.current_trace_height()
    }

    fn trace_width(&self) -> usize {
        self.0.trace_width()
    }
}

impl<RA> Chip<RA, GpuBackend> for DeviceProgramTester {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let height = next_power_of_two_or_zero(self.0.current_trace_height());
        let width = self.0.trace_width();

        if height == 0 {
            return get_empty_air_proving_ctx();
        }
        let trace = DeviceMatrix::<F>::with_capacity(height, width);

        let records = &self.0.records;
        let num_records = records.len();

        unsafe {
            let bytes_size = num_records * size_of::<ProgramExecutionCols<F>>();
            let records_bytes = from_raw_parts(records.as_ptr() as *const u8, bytes_size);
            let records = records_bytes.to_device().unwrap();
            program_testing::tracegen(trace.buffer(), height, width, &records, num_records)
                .unwrap();
        }
        AirProvingContext::simple_no_pis(trace)
    }
}
