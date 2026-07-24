use std::{mem::size_of, sync::Arc};

use openvm_circuit::{primitives::Chip, system::program::ProgramExecutionCols};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend, GpuDevice};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, pinned, stream::GpuDeviceCtx};
use openvm_instructions::{program::Program, LocalOpcode, SystemOpcode};
use openvm_stark_backend::prover::{
    AirProvingContext, CommittedTraceData, MatrixDimensions, TraceCommitter,
};
use p3_field::PrimeCharacteristicRing;

use super::memory::DeviceInitialMemory;
use crate::cuda_abi::program;

/// Gap-filtered program execution frequencies reconstructed on device from
/// the native preflight block-run stream.
pub struct DeviceProgramFrequencies {
    pub frequencies: DeviceBuffer<u32>,
}

/// Optional producer for the CUDA all-direct chronology route. The initial
/// memory descriptors are passed through because the shared RVR predecode
/// reconstructs memory and program chronology in one launch sequence.
pub trait DeviceProgramFrequenciesProvider: Send + Sync {
    fn take_device_program_frequencies(
        &self,
        device_ctx: &GpuDeviceCtx,
        initial_memory: &[DeviceInitialMemory],
    ) -> Option<DeviceProgramFrequencies>;
}

pub struct ProgramChipGPU {
    pub cached: Option<CommittedTraceData<GpuBackend>>,
    pub device_ctx: GpuDeviceCtx,
}

impl ProgramChipGPU {
    pub fn new(device_ctx: GpuDeviceCtx) -> Self {
        Self {
            cached: None,
            device_ctx,
        }
    }

    pub fn generate_cached_trace(
        program: Program<F>,
        device_ctx: &GpuDeviceCtx,
    ) -> DeviceMatrix<F> {
        let instructions = program
            .enumerate_by_pc()
            .into_iter()
            .map(|(pc, instruction, _)| {
                [
                    F::from_u32(pc),
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
        let height = num_records.next_power_of_two();
        let records = instructions
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .to_device_on(device_ctx)
            .unwrap();

        let trace = DeviceMatrix::<F>::with_capacity_on(
            height,
            size_of::<ProgramExecutionCols<u8>>(),
            device_ctx,
        );
        trace.buffer().fill_zero_on(device_ctx).unwrap();
        unsafe {
            program::cached_tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &records,
                program.pc_base,
                SystemOpcode::TERMINATE.global_opcode().as_usize(),
                device_ctx.stream.as_raw(),
            )
            .expect("Failed to generate cached trace");
        }
        trace
    }

    pub fn get_committed_trace(
        trace: DeviceMatrix<F>,
        device: &GpuDevice,
    ) -> CommittedTraceData<GpuBackend> {
        let (commitment, data) = TraceCommitter::<GpuBackend>::commit(device, &[&trace]).unwrap();
        CommittedTraceData {
            commitment,
            data: Arc::new(data),
            trace,
        }
    }

    pub(crate) fn generate_proving_ctx_from_frequencies(
        &self,
        filtered_exec_freqs: &[u32],
    ) -> AirProvingContext<GpuBackend> {
        let cached = self.cached.clone().expect("Cached program must be loaded");
        let height = cached.height();
        let filtered_len = filtered_exec_freqs.len();
        assert!(
            filtered_len <= height,
            "filtered_exec_freqs len={filtered_len} > cached trace height={height}"
        );
        let buffer: DeviceBuffer<F> = DeviceBuffer::with_capacity_on(height, &self.device_ctx);

        // Upload the raw u32 frequencies through a pooled pinned buffer and
        // convert to field elements on device (also zero-filling the tail).
        let bytes = std::mem::size_of_val(filtered_exec_freqs);
        let mut h_freqs = pinned::take(bytes + std::mem::size_of::<u32>());
        let off = h_freqs.as_ptr().align_offset(std::mem::size_of::<u32>());
        // SAFETY: the ranges are in-bounds, disjoint allocations, and the
        // destination is 4-aligned by `off`.
        let words: &[u32] = unsafe {
            std::ptr::copy_nonoverlapping(
                filtered_exec_freqs.as_ptr() as *const u8,
                h_freqs.as_mut_ptr().add(off),
                bytes,
            );
            std::slice::from_raw_parts(h_freqs.as_ptr().add(off) as *const u32, filtered_len)
        };
        let d_freqs = words
            .to_device_on(&self.device_ctx)
            .expect("failed to copy exec frequencies to device");
        pinned::give_back(h_freqs, off + bytes);
        unsafe {
            crate::cuda_abi::program::fill_frequencies(
                &d_freqs,
                filtered_len,
                &buffer,
                height,
                self.device_ctx.stream.as_raw(),
            )
            .expect("program_fill_frequencies failed");
        }

        let common_main = DeviceMatrix::new(Arc::new(buffer), height, 1);

        AirProvingContext {
            cached_mains: vec![cached],
            common_main,
            public_values: vec![],
        }
    }

    pub(crate) fn generate_proving_ctx_from_device_frequencies(
        &self,
        frequencies: DeviceProgramFrequencies,
    ) -> AirProvingContext<GpuBackend> {
        let cached = self.cached.clone().expect("Cached program must be loaded");
        let height = cached.height();
        assert!(
            frequencies.frequencies.len() <= height,
            "device program frequencies len={} > cached trace height={height}",
            frequencies.frequencies.len()
        );
        let buffer = DeviceBuffer::<F>::with_capacity_on(height, &self.device_ctx);
        unsafe {
            program::frequency_tracegen(
                &buffer,
                height,
                &frequencies.frequencies,
                self.device_ctx.stream.as_raw(),
            )
            .expect("Failed to generate device program-frequency trace");
        }
        let common_main = DeviceMatrix::new(Arc::new(buffer), height, 1);
        AirProvingContext {
            cached_mains: vec![cached],
            common_main,
            public_values: vec![],
        }
    }
}

impl Default for ProgramChipGPU {
    fn default() -> Self {
        panic!("ProgramChipGPU requires an explicit GpuDeviceCtx")
    }
}

impl Chip<Vec<u32>, GpuBackend> for ProgramChipGPU {
    fn generate_proving_ctx(&self, filtered_exec_freqs: Vec<u32>) -> AirProvingContext<GpuBackend> {
        self.generate_proving_ctx_from_frequencies(&filtered_exec_freqs)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openvm_cuda_backend::{data_transporter::assert_eq_host_and_device_matrix, prelude::F};
    use openvm_instructions::{
        instruction::Instruction,
        program::{Program, DEFAULT_PC_STEP},
        LocalOpcode,
        SystemOpcode::*,
    };
    use openvm_stark_backend::{prover::TraceCommitter, StarkEngine};

    use super::ProgramChipGPU;
    use crate::{
        system::program::{
            tests::{BEQ, BNE, JAL, STOREW, SUB},
            trace::generate_cached_trace,
        },
        utils::{test_cpu_engine, test_gpu_engine},
    };

    fn test_cached_committed_trace_data(program: Program<F>) {
        let gpu_engine = test_gpu_engine();
        let gpu_device = gpu_engine.device();
        let gpu_trace =
            ProgramChipGPU::generate_cached_trace(program.clone(), &gpu_device.device_ctx);
        let gpu_cached = ProgramChipGPU::get_committed_trace(gpu_trace, gpu_device);

        let cpu_engine = test_cpu_engine();
        let cpu_device = cpu_engine.device();
        let cpu_trace = Arc::new(generate_cached_trace(&program));
        let (cpu_commit, _) = cpu_device.commit(&[&cpu_trace]).unwrap();

        // NOTE: This compares the stacked matrices, not the original cached trace
        assert_eq_host_and_device_matrix(cpu_trace, &gpu_cached.trace, &gpu_device.device_ctx);
        assert_eq!(gpu_cached.commitment, cpu_commit);
    }

    #[test]
    fn test_cuda_program_cached_tracegen_1() {
        let instructions = vec![
            Instruction::large_from_isize(STOREW, 2, 0, 0, 0, 1, 0, 1),
            Instruction::large_from_isize(STOREW, 1, 1, 0, 0, 1, 0, 1),
            Instruction::from_isize(BEQ, 0, 0, 3 * DEFAULT_PC_STEP as isize, 1, 0),
            Instruction::from_isize(SUB, 0, 0, 1, 1, 1),
            Instruction::from_isize(JAL, 2, -2 * (DEFAULT_PC_STEP as isize), 0, 1, 0),
            Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let program = Program::from_instructions(&instructions);
        test_cached_committed_trace_data(program);
    }

    #[test]
    fn test_cuda_program_cached_tracegen_2() {
        let instructions = vec![
            Instruction::large_from_isize(STOREW, 5, 0, 0, 0, 1, 0, 1),
            Instruction::from_isize(BNE, 0, 4, 3 * DEFAULT_PC_STEP as isize, 1, 0),
            Instruction::from_isize(JAL, 2, -2 * DEFAULT_PC_STEP as isize, 0, 1, 0),
            Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
            Instruction::from_isize(BEQ, 0, 5, -(DEFAULT_PC_STEP as isize), 1, 0),
        ];
        let program = Program::from_instructions(&instructions);
        test_cached_committed_trace_data(program);
    }

    #[test]
    fn test_cuda_program_cached_tracegen_undefined_instructions() {
        let instructions = vec![
            Some(Instruction::large_from_isize(STOREW, 2, 0, 0, 0, 1, 0, 1)),
            Some(Instruction::large_from_isize(STOREW, 1, 1, 0, 0, 1, 0, 1)),
            Some(Instruction::from_isize(
                BEQ,
                0,
                2,
                3 * DEFAULT_PC_STEP as isize,
                1,
                0,
            )),
            None,
            None,
            Some(Instruction::from_isize(
                TERMINATE.global_opcode(),
                0,
                0,
                0,
                0,
                0,
            )),
        ];
        let program = Program::new_without_debug_infos_with_option(&instructions, 0);
        test_cached_committed_trace_data(program);
    }
}
