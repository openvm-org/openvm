use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::var_range::cuda::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use super::{FieldArithmeticCoreCols, FieldArithmeticRecord};
use crate::{
    adapters::{AluNativeAdapterCols, AluNativeAdapterRecord},
    cuda_abi::field_arithmetic_cuda,
};

#[derive(new)]
pub struct FieldArithmeticChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for FieldArithmeticChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize =
            size_of::<(AluNativeAdapterRecord<F>, FieldArithmeticRecord<F>)>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        assert_eq!(records.len() % RECORD_SIZE, 0);

        let height = records.len() / RECORD_SIZE;
        let padded_height = next_power_of_two_or_zero(height);
        let trace_width =
            AluNativeAdapterCols::<F>::width() + FieldArithmeticCoreCols::<F>::width();
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, trace_width);

        let d_records = records.to_device().unwrap();

        unsafe {
            field_arithmetic_cuda::tracegen(
                trace.buffer(),
                padded_height,
                trace_width,
                &d_records,
                self.range_checker.count.as_ptr() as *const u32,
                self.range_checker.count.len(),
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::{AluNativeAdapterAir, AluNativeAdapterExecutor, AluNativeAdapterFiller};
    use crate::field_arithmetic::{
        FieldArithmeticAir, FieldArithmeticChip, FieldArithmeticCoreAir, FieldArithmeticCoreFiller,
        FieldArithmeticExecutor,
    };
    use crate::write_native_array;
    use openvm_circuit::arch::testing::GpuChipTestBuilder;
    use openvm_circuit::arch::testing::GpuTestChipHarness;
    use openvm_circuit::arch::testing::TestBuilder;
    use openvm_circuit::arch::{testing::memory::gen_pointer, EmptyAdapterCoreLayout};
    use openvm_instructions::{instruction::Instruction, LocalOpcode};
    use openvm_native_compiler::{conversion::AS, FieldArithmeticOpcode};
    use openvm_stark_backend::p3_field::{Field, FieldAlgebra};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::Rng;
    use test_case::test_case;

    const MAX_INS_CAPACITY: usize = 128;

    fn create_test_harness(
        tester: &GpuChipTestBuilder,
    ) -> GpuTestChipHarness<
        F,
        FieldArithmeticExecutor,
        FieldArithmeticAir,
        FieldArithmeticChipGpu,
        FieldArithmeticChip<F>,
    > {
        let adapter_air =
            AluNativeAdapterAir::new(tester.execution_bridge(), tester.memory_bridge());
        let core_air = FieldArithmeticCoreAir::new();
        let air = FieldArithmeticAir::new(adapter_air, core_air);

        let adapter_step = AluNativeAdapterExecutor::new();
        let executor = FieldArithmeticExecutor::new(adapter_step);

        let core_filler = FieldArithmeticCoreFiller::new(AluNativeAdapterFiller);

        let cpu_chip = FieldArithmeticChip::new(core_filler, tester.dummy_memory_helper());
        let gpu_chip =
            FieldArithmeticChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    #[test_case(FieldArithmeticOpcode::ADD, 100)]
    #[test_case(FieldArithmeticOpcode::SUB, 100)]
    #[test_case(FieldArithmeticOpcode::MUL, 100)]
    #[test_case(FieldArithmeticOpcode::DIV, 100)]
    fn rand_field_arithmetic_tracegen_test(opcode: FieldArithmeticOpcode, num_ops: usize) {
        let mut tester = GpuChipTestBuilder::default();
        let mut rng = create_seeded_rng();

        let mut harness = create_test_harness(&tester);

        for _ in 0..num_ops {
            let b_val = rng.gen::<F>();
            let c_val = if opcode == FieldArithmeticOpcode::DIV {
                loop {
                    let x = rng.gen::<F>();
                    if !x.is_zero() {
                        break x;
                    }
                }
            } else {
                rng.gen::<F>()
            };

            let (_, b_ptr) = write_native_array(&mut tester, &mut rng, Some([b_val]));
            let (_, c_ptr) = write_native_array(&mut tester, &mut rng, Some([c_val]));

            let a = gen_pointer(&mut rng, 1);
            tester.execute(
                &mut harness.executor,
                &mut harness.dense_arena,
                &Instruction::new(
                    opcode.global_opcode(),
                    F::from_canonical_usize(a),
                    F::from_canonical_usize(b_ptr),
                    F::from_canonical_usize(c_ptr),
                    F::from_canonical_usize(AS::Native as usize),
                    F::from_canonical_usize(AS::Native as usize),
                    F::from_canonical_usize(AS::Native as usize),
                    F::ZERO,
                ),
            );
        }

        type Record<'a> = (
            &'a mut AluNativeAdapterRecord<F>,
            &'a mut FieldArithmeticRecord<F>,
        );

        harness
            .dense_arena
            .get_record_seeker::<Record<'_>, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, AluNativeAdapterExecutor>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
