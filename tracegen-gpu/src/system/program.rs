use std::{mem::size_of, sync::Arc};

use openvm_circuit::{
    system::program::{ProgramAir, ProgramExecutionCols},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::{program::Program, LocalOpcode, SystemOpcode};
use openvm_stark_backend::{
    p3_field::FieldAlgebra, prover::hal::MatrixDimensions, rap::get_air_name, AirRef,
    ChipUsageGetter,
};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::copy::MemCopyH2D,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::{system::cuda, DeviceChip};

pub struct ProgramChipGpu {
    pub air: ProgramAir,
    pub cached_trace: Option<DeviceMatrix<F>>,
    pub exec_freqs: Vec<u32>,
}

impl ProgramChipGpu {
    pub fn new(air: ProgramAir) -> Self {
        Self {
            air,
            cached_trace: None,
            exec_freqs: Vec::new(),
        }
    }

    pub fn generate_cached_trace(&mut self, program: Program<F>) {
        assert!(self.cached_trace.is_none());
        let instructions = program
            .enumerate_by_pc()
            .into_iter()
            .map(|(pc, instruction, _)| {
                [
                    F::from_canonical_u32(pc),
                    instruction.opcode.to_field(),
                    instruction.a,
                    instruction.b,
                    instruction.c,
                    instruction.d,
                    instruction.e,
                    instruction.f,
                    instruction.g,
                ]
            })
            .collect::<Vec<_>>();

        let num_records = instructions.len();
        let height = next_power_of_two_or_zero(num_records);
        let records = instructions
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .to_device()
            .unwrap();

        let trace = DeviceMatrix::<F>::with_capacity(height, size_of::<ProgramExecutionCols<u8>>());
        unsafe {
            cuda::program::cached_tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &records,
                num_records,
                program.pc_base + num_records as u32 * program.step,
                SystemOpcode::TERMINATE.global_opcode().as_usize(),
            )
            .expect("Failed to generate cached trace");
        }
        self.cached_trace = Some(trace);
    }
}

impl ChipUsageGetter for ProgramChipGpu {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.cached_trace.as_ref().map(|t| t.height()).unwrap_or(0)
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for ProgramChipGpu {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    // TODO[stephen]: This currently generates the common main trace, but with the
    // changes introduced in OpenVM's feat/new-exec-device this will generate the
    // whole proving context (i.e. cached trace + common main).
    fn generate_trace(&self) -> DeviceMatrix<F> {
        let height = self.current_trace_height();
        assert!(height > 0);
        assert!(self.exec_freqs.len() == height);
        let buffer = self
            .exec_freqs
            .iter()
            .map(|f| F::from_canonical_u32(*f))
            .collect::<Vec<_>>()
            .to_device()
            .unwrap();
        DeviceMatrix::new(Arc::new(buffer), height, 1)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_program_tracegen() {
        // TODO[stephen]: test ProgramChipGpu tracegen after feat/new-exec-device
        println!("Skipping test_program_tracegen...");
    }
}
