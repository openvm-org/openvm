use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_native_circuit::{
    adapters::{BranchNativeAdapterCols, BranchNativeAdapterRecord},
    NativeBranchEqualCoreRecord,
};
use openvm_rv32im_circuit::BranchEqualCoreCols;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prover_backend::GpuBackend, types::F,
};

use crate::{
    extensions::native::native_branch_eq_cuda, get_empty_air_proving_ctx,
    primitives::var_range::VariableRangeCheckerChipGPU,
};

#[derive(new)]
pub struct NativeBranchEqChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for NativeBranchEqChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize =
            size_of::<(BranchNativeAdapterRecord<F>, NativeBranchEqualCoreRecord<F>)>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        assert_eq!(records.len() % RECORD_SIZE, 0);

        let height = records.len() / RECORD_SIZE;
        let padded_height = next_power_of_two_or_zero(height);
        let trace_width =
            BranchNativeAdapterCols::<F>::width() + BranchEqualCoreCols::<F, 1>::width();
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, trace_width);

        let d_records = records.to_device().unwrap();

        unsafe {
            native_branch_eq_cuda::tracegen(
                trace.buffer(),
                padded_height as u32,
                trace_width as u32,
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
    use openvm_circuit::arch::EmptyAdapterCoreLayout;
    use openvm_instructions::{
        instruction::Instruction,
        program::{DEFAULT_PC_STEP, PC_BITS},
        utils::isize_to_field,
        LocalOpcode,
    };
    use openvm_native_circuit::{
        adapters::{
            BranchNativeAdapterAir, BranchNativeAdapterExecutor, BranchNativeAdapterFiller,
        },
        NativeBranchEqAir, NativeBranchEqChip, NativeBranchEqExecutor, NativeBranchEqualFiller,
    };
    use openvm_native_compiler::NativeBranchEqualOpcode;
    use openvm_rv32im_circuit::{adapters::RV_B_TYPE_IMM_BITS, BranchEqualCoreAir};
    use openvm_rv32im_transpiler::BranchEqualOpcode;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use test_case::test_case;

    use super::*;
    use crate::{
        extensions::native::write_native_or_imm,
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
    };

    const ABS_MAX_IMM: i32 = 1 << (RV_B_TYPE_IMM_BITS - 1);
    const MAX_INS_CAPACITY: usize = 128;

    fn create_test_harness(
        tester: &GpuChipTestBuilder,
    ) -> GpuTestChipHarness<
        F,
        NativeBranchEqExecutor,
        NativeBranchEqAir,
        NativeBranchEqChipGpu,
        NativeBranchEqChip<F>,
    > {
        let adapter_air =
            BranchNativeAdapterAir::new(tester.execution_bridge(), tester.memory_bridge());
        let core_air =
            BranchEqualCoreAir::new(NativeBranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP);
        let air = NativeBranchEqAir::new(adapter_air, core_air);

        let adapter_step = BranchNativeAdapterExecutor::new();
        let executor = NativeBranchEqExecutor::new(
            adapter_step,
            NativeBranchEqualOpcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        );

        let core_filler = NativeBranchEqualFiller::new(BranchNativeAdapterFiller);

        let cpu_chip = NativeBranchEqChip::new(core_filler, tester.dummy_memory_helper());
        let gpu_chip =
            NativeBranchEqChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        harness: &mut GpuTestChipHarness<
            F,
            NativeBranchEqExecutor,
            NativeBranchEqAir,
            NativeBranchEqChipGpu,
            NativeBranchEqChip<F>,
        >,
        rng: &mut StdRng,
        opcode: NativeBranchEqualOpcode,
    ) {
        let a_val = rng.gen();
        let b_val = if rng.gen_bool(0.5) { a_val } else { rng.gen() };
        let imm = rng.gen_range((-ABS_MAX_IMM)..ABS_MAX_IMM);
        let (a, a_as) = write_native_or_imm(tester, rng, a_val, None);
        let (b, b_as) = write_native_or_imm(tester, rng, b_val, None);
        let initial_pc =
            rng.gen_range(imm.unsigned_abs()..(1 << (PC_BITS - 1)) - imm.unsigned_abs());

        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            &Instruction::new(
                opcode.global_opcode(),
                a,
                b,
                isize_to_field::<F>(imm as isize),
                F::from_canonical_usize(a_as),
                F::from_canonical_usize(b_as),
                F::ZERO,
                F::ZERO,
            ),
            initial_pc,
        );
    }

    #[test_case(BranchEqualOpcode::BEQ, 100)]
    #[test_case(BranchEqualOpcode::BNE, 100)]
    fn test_native_branch_eq_tracegen(opcode: BranchEqualOpcode, num_ops: usize) {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default();

        let mut harness = create_test_harness(&tester);

        let opcode = NativeBranchEqualOpcode(opcode);
        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut harness, &mut rng, opcode);
        }

        type Record<'a> = (
            &'a mut BranchNativeAdapterRecord<F>,
            &'a mut NativeBranchEqualCoreRecord<F>,
        );

        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, BranchNativeAdapterExecutor>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
