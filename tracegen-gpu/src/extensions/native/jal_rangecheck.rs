use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_native_circuit::{JalRangeCheckAir, JalRangeCheckRecord};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::copy::MemCopyH2D,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::{
    extensions::native::native_jal_rangecheck_cuda,
    primitives::var_range::VariableRangeCheckerChipGPU, DeviceChip,
};

#[derive(new)]
pub struct JalRangeCheckGpu {
    pub air: JalRangeCheckAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub arena: DenseRecordArena,
}

impl ChipUsageGetter for JalRangeCheckGpu {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        let record_size = size_of::<JalRangeCheckRecord<F>>();
        let records_len = self.arena.allocated().len();
        assert_eq!(records_len % record_size, 0);
        records_len / record_size
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for JalRangeCheckGpu {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records = self.arena.allocated().to_device().unwrap();
        let height = self.current_trace_height();
        let padded_height = next_power_of_two_or_zero(height);
        let width = self.trace_width();
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, width);

        unsafe {
            native_jal_rangecheck_cuda::tracegen(
                trace.buffer(),
                padded_height as u32,
                width as u32,
                &d_records,
                height as u32,
                &self.range_checker.count,
            )
            .unwrap();
        }

        trace
    }
}

#[cfg(test)]
mod test {
    use openvm_circuit::arch::{EmptyMultiRowLayout, NewVmChipWrapper};
    use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode, VmOpcode};
    use openvm_native_circuit::{JalRangeCheckChip, JalRangeCheckStep};
    use openvm_native_compiler::{conversion::AS, NativeJalOpcode, NativeRangeCheckOpcode};
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::Rng;
    use test_case::test_case;

    use super::*;
    use crate::testing::GpuChipTestBuilder;

    const MAX_INS_CAPACITY: usize = 128;
    type DenseChip<F> = NewVmChipWrapper<F, JalRangeCheckAir, JalRangeCheckStep, DenseRecordArena>;
    type SparseChip<F> = JalRangeCheckChip<F>;

    fn create_test_dense_chip(tester: &GpuChipTestBuilder) -> DenseChip<F> {
        let range_checker = tester.cpu_range_checker().clone();
        let mut chip = DenseChip::<F>::new(
            JalRangeCheckAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                range_checker.bus(),
            ),
            JalRangeCheckStep::new(range_checker),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_test_sparse_chip(tester: &GpuChipTestBuilder) -> SparseChip<F> {
        let range_checker = tester.cpu_range_checker().clone();
        let mut chip = SparseChip::<F>::new(
            JalRangeCheckAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                range_checker.bus(),
            ),
            JalRangeCheckStep::new(range_checker),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    #[test_case(NativeJalOpcode::JAL.global_opcode(), 100)]
    #[test_case(NativeRangeCheckOpcode::RANGE_CHECK.global_opcode(), 100)]
    fn test_jal_rangecheck_tracegen(opcode: VmOpcode, num_ops: usize) {
        use openvm_circuit::arch::testing::memory::gen_pointer;

        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default().with_variable_range_checker();
        let mut dense_chip = create_test_dense_chip(&tester);

        for _ in 0..num_ops {
            if opcode == NativeJalOpcode::JAL.global_opcode() {
                let initial_pc = rng.gen_range(0..(1 << PC_BITS));
                let a = gen_pointer(&mut rng, 1);
                let b = rng.gen_range(0..(1 << 16));

                tester.execute_with_pc(
                    &mut dense_chip,
                    &Instruction::from_usize(
                        opcode,
                        [a, b as usize, 0, AS::Native as usize, 0, 0, 0],
                    ),
                    initial_pc,
                );
            } else {
                // RANGE_CHECK
                let a_val = rng.gen_range(0..(1 << 30));
                let a = gen_pointer(&mut rng, 1);
                tester.write::<1>(AS::Native as usize, a, [F::from_canonical_u32(a_val)]);

                let x = a_val & 0xffff;
                let y = a_val >> 16;
                let min_b = 32 - x.leading_zeros();
                let min_c = 32 - y.leading_zeros();
                let b = rng.gen_range(min_b..=16);
                let c = rng.gen_range(min_c..=14);

                tester.execute(
                    &mut dense_chip,
                    &Instruction::from_usize(
                        opcode,
                        [a, b as usize, c as usize, AS::Native as usize, 0, 0, 0],
                    ),
                );
            }
        }

        let mut sparse_chip = create_test_sparse_chip(&tester);

        type Record<'a> = &'a mut JalRangeCheckRecord<F>;

        dense_chip
            .arena
            .get_record_seeker::<Record<'_>, EmptyMultiRowLayout>()
            .transfer_to_matrix_arena(&mut sparse_chip.arena);

        let gpu_chip =
            JalRangeCheckGpu::new(dense_chip.air, tester.range_checker(), dense_chip.arena);

        tester
            .build()
            .load_and_compare(gpu_chip, sparse_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
