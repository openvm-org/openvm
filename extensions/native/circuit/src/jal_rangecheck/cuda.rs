use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::var_range::cuda::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use super::{JalRangeCheckCols, JalRangeCheckRecord};
use crate::cuda_abi::native_jal_rangecheck_cuda;

#[derive(new)]
pub struct JalRangeCheckGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for JalRangeCheckGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<JalRangeCheckRecord<F>>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        assert_eq!(records.len() % RECORD_SIZE, 0);

        let width = JalRangeCheckCols::<F>::width();

        let height = records.len() / RECORD_SIZE;
        let padded_height = next_power_of_two_or_zero(height);
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, width);

        let d_records = records.to_device().unwrap();

        unsafe {
            native_jal_rangecheck_cuda::tracegen(
                trace.buffer(),
                padded_height as u32,
                width as u32,
                &d_records,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(trace)
    }
}

#[cfg(test)]
mod test {
    use crate::jal_rangecheck::{
        JalRangeCheckAir, JalRangeCheckExecutor, JalRangeCheckFiller, NativeJalRangeCheckChip,
    };
    use openvm_circuit::arch::testing::GpuChipTestBuilder;
    use openvm_circuit::arch::testing::GpuTestChipHarness;
    use openvm_circuit::arch::testing::TestBuilder;
    use openvm_circuit::arch::{testing::memory::gen_pointer, EmptyMultiRowLayout};
    use openvm_circuit::utils::cuda_utils::default_var_range_checker_bus;
    use openvm_circuit::utils::cuda_utils::dummy_range_checker;
    use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode, VmOpcode};
    use openvm_native_compiler::{conversion::AS, NativeJalOpcode, NativeRangeCheckOpcode};
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::Rng;
    use test_case::test_case;

    use super::*;

    const MAX_INS_CAPACITY: usize = 128;

    fn create_test_harness(
        tester: &GpuChipTestBuilder,
    ) -> GpuTestChipHarness<
        F,
        JalRangeCheckExecutor,
        JalRangeCheckAir,
        JalRangeCheckGpu,
        NativeJalRangeCheckChip<F>,
    > {
        let range_bus = default_var_range_checker_bus();
        let air =
            JalRangeCheckAir::new(tester.execution_bridge(), tester.memory_bridge(), range_bus);
        let executor = JalRangeCheckExecutor::new();
        let filler = JalRangeCheckFiller::new(dummy_range_checker(range_bus));
        let cpu_chip = NativeJalRangeCheckChip::<F>::new(filler, tester.dummy_memory_helper());
        let gpu_chip = JalRangeCheckGpu::new(tester.range_checker(), tester.timestamp_max_bits());

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    #[test_case(NativeJalOpcode::JAL.global_opcode(), 100)]
    #[test_case(NativeRangeCheckOpcode::RANGE_CHECK.global_opcode(), 100)]
    fn test_jal_rangecheck_tracegen(opcode: VmOpcode, num_ops: usize) {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default();
        let mut harness = create_test_harness(&tester);

        for _ in 0..num_ops {
            if opcode == NativeJalOpcode::JAL.global_opcode() {
                let initial_pc = rng.gen_range(0..(1 << PC_BITS));
                let a = gen_pointer(&mut rng, 1);
                let b = rng.gen_range(0..(1 << 16));

                tester.execute_with_pc(
                    &mut harness.executor,
                    &mut harness.dense_arena,
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
                    &mut harness.executor,
                    &mut harness.dense_arena,
                    &Instruction::from_usize(
                        opcode,
                        [a, b as usize, c as usize, AS::Native as usize, 0, 0, 0],
                    ),
                );
            }
        }

        type Record<'a> = &'a mut JalRangeCheckRecord<F>;

        harness
            .dense_arena
            .get_record_seeker::<Record<'_>, EmptyMultiRowLayout>()
            .transfer_to_matrix_arena(&mut harness.matrix_arena);

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
