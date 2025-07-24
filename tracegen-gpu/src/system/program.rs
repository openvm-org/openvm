use std::{mem::size_of, sync::Arc};

use openvm_circuit::{system::program::ProgramExecutionCols, utils::next_power_of_two_or_zero};
use openvm_instructions::{program::Program, LocalOpcode, SystemOpcode};
use openvm_stark_backend::{
    p3_field::FieldAlgebra,
    prover::{
        hal::{MatrixDimensions, TraceCommitter},
        types::{AirProvingContext, CommittedTraceData},
    },
    Chip,
};
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer},
    gpu_device::GpuDevice,
    prover_backend::GpuBackend,
    types::F,
};

use crate::system::cuda;

pub struct ProgramChipGPU {
    pub cached: Option<CommittedTraceData<GpuBackend>>,
    pub filtered_exec_freqs: Vec<u32>,
}

impl ProgramChipGPU {
    pub fn new() -> Self {
        Self {
            cached: None,
            filtered_exec_freqs: Vec::new(),
        }
    }

    pub fn generate_cached_trace(program: Program<F>) -> DeviceMatrix<F> {
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
                program.pc_base,
                program.step,
                SystemOpcode::TERMINATE.global_opcode().as_usize(),
            )
            .expect("Failed to generate cached trace");
        }
        trace
    }

    pub fn get_committed_trace(
        trace: DeviceMatrix<F>,
        device: &GpuDevice,
    ) -> CommittedTraceData<GpuBackend> {
        let (root, pcs_data) = device.commit(&[trace.clone()]);
        CommittedTraceData {
            commitment: root,
            trace,
            data: pcs_data,
        }
    }
}

impl Default for ProgramChipGPU {
    fn default() -> Self {
        Self::new()
    }
}

impl<RA> Chip<RA, GpuBackend> for ProgramChipGPU {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let cached = self.cached.clone().expect("Cached program must be loaded");
        let height = cached.trace.height();
        assert!(
            self.filtered_exec_freqs.len() <= height,
            "filtered_exec_freqs len={} > cached trace height={}",
            self.filtered_exec_freqs.len(),
            height
        );
        let mut buffer: DeviceBuffer<F> = DeviceBuffer::with_capacity(height);

        // Making sure to zero-out the untouched part of the buffer.
        if self.filtered_exec_freqs.len() < height {
            buffer
                .fill_zero_suffix(self.filtered_exec_freqs.len())
                .unwrap();
        }

        self.filtered_exec_freqs
            .iter()
            .map(|f| F::from_canonical_u32(*f))
            .collect::<Vec<_>>()
            .copy_to(&mut buffer)
            .unwrap();

        let trace = DeviceMatrix::new(Arc::new(buffer), height, 1);

        AirProvingContext {
            cached_mains: vec![cached],
            common_main: Some(trace),
            public_values: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use openvm_circuit::system::program::trace::VmCommittedExe;
    use openvm_instructions::{
        exe::VmExe,
        instruction::Instruction,
        program::{Program, DEFAULT_PC_STEP},
        LocalOpcode,
        SystemOpcode::*,
    };
    use openvm_native_compiler::{
        FieldArithmeticOpcode::*, NativeBranchEqualOpcode, NativeJalOpcode::*,
        NativeLoadStoreOpcode::*,
    };
    use openvm_rv32im_transpiler::BranchEqualOpcode::*;
    use openvm_stark_backend::config::StarkGenericConfig;
    use openvm_stark_sdk::{
        config::{
            baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
            FriParameters,
        },
        engine::{StarkEngine, StarkFriEngine},
    };
    use stark_backend_gpu::{engine::GpuBabyBearPoseidon2Engine, types::F};

    use crate::{system::program::ProgramChipGPU, testing::assert_eq_cpu_and_gpu_matrix};

    fn test_cached_committed_trace_data(program: Program<F>) {
        let gpu_engine = GpuBabyBearPoseidon2Engine::new(FriParameters::new_for_testing(2));
        let gpu_device = gpu_engine.device();
        let gpu_trace = ProgramChipGPU::generate_cached_trace(program.clone());
        let gpu_cached = ProgramChipGPU::get_committed_trace(gpu_trace, gpu_device);

        let cpu_engine = BabyBearPoseidon2Engine::new(FriParameters::new_for_testing(2));
        let cpu_exe = VmExe::new(program.clone());
        let cpu_committed_exe =
            VmCommittedExe::<BabyBearPoseidon2Config>::commit(cpu_exe, cpu_engine.config().pcs());
        let cpu_cached = cpu_committed_exe.get_committed_trace();

        assert_eq_cpu_and_gpu_matrix(cpu_cached.trace, &gpu_cached.trace);
        assert_eq!(gpu_cached.commitment, cpu_cached.commitment);
    }

    #[test]
    fn test_program_cached_tracegen_1() {
        let instructions = vec![
            Instruction::large_from_isize(STOREW.global_opcode(), 2, 0, 0, 0, 1, 0, 1),
            Instruction::large_from_isize(STOREW.global_opcode(), 1, 1, 0, 0, 1, 0, 1),
            Instruction::from_isize(
                NativeBranchEqualOpcode(BEQ).global_opcode(),
                0,
                0,
                3 * DEFAULT_PC_STEP as isize,
                1,
                0,
            ),
            Instruction::from_isize(SUB.global_opcode(), 0, 0, 1, 1, 1),
            Instruction::from_isize(
                JAL.global_opcode(),
                2,
                -2 * (DEFAULT_PC_STEP as isize),
                0,
                1,
                0,
            ),
            Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let program = Program::from_instructions(&instructions);
        test_cached_committed_trace_data(program);
    }

    #[test]
    fn test_program_cached_tracegen_2() {
        let instructions = vec![
            Instruction::large_from_isize(STOREW.global_opcode(), 5, 0, 0, 0, 1, 0, 1),
            Instruction::from_isize(
                NativeBranchEqualOpcode(BNE).global_opcode(),
                0,
                4,
                3 * DEFAULT_PC_STEP as isize,
                1,
                0,
            ),
            Instruction::from_isize(
                JAL.global_opcode(),
                2,
                -2 * DEFAULT_PC_STEP as isize,
                0,
                1,
                0,
            ),
            Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
            Instruction::from_isize(
                NativeBranchEqualOpcode(BEQ).global_opcode(),
                0,
                5,
                -(DEFAULT_PC_STEP as isize),
                1,
                0,
            ),
        ];
        let program = Program::from_instructions(&instructions);
        test_cached_committed_trace_data(program);
    }

    #[test]
    fn test_program_cached_tracegen_undefined_instructions() {
        let instructions = vec![
            Some(Instruction::large_from_isize(
                STOREW.global_opcode(),
                2,
                0,
                0,
                0,
                1,
                0,
                1,
            )),
            Some(Instruction::large_from_isize(
                STOREW.global_opcode(),
                1,
                1,
                0,
                0,
                1,
                0,
                1,
            )),
            Some(Instruction::from_isize(
                NativeBranchEqualOpcode(BEQ).global_opcode(),
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
        let program =
            Program::new_without_debug_infos_with_option(&instructions, DEFAULT_PC_STEP, 0);
        test_cached_committed_trace_data(program);
    }
}
