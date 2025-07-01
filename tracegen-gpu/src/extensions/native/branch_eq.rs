use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_native_circuit::{
    adapters::BranchNativeAdapterRecord, NativeBranchEqAir, NativeBranchEqualCoreRecord,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::copy::MemCopyH2D,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::{
    extensions::native::native_branch_eq_cuda, primitives::var_range::VariableRangeCheckerChipGPU,
    DeviceChip,
};

#[derive(new)]
pub struct NativeBranchEqChipGpu {
    pub air: NativeBranchEqAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub arena: DenseRecordArena,
}

impl ChipUsageGetter for NativeBranchEqChipGpu {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize =
            size_of::<(BranchNativeAdapterRecord<F>, NativeBranchEqualCoreRecord<F>)>();
        let records_len = self.arena.allocated().len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for NativeBranchEqChipGpu {
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
            native_branch_eq_cuda::tracegen(
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
    use openvm_circuit::arch::{EmptyAdapterCoreLayout, MatrixRecordArena, NewVmChipWrapper};
    use openvm_instructions::{
        instruction::Instruction,
        program::{DEFAULT_PC_STEP, PC_BITS},
        utils::isize_to_field,
        LocalOpcode,
    };
    use openvm_native_circuit::{
        adapters::{BranchNativeAdapterAir, BranchNativeAdapterStep},
        NativeBranchEqStep,
    };
    use openvm_native_compiler::NativeBranchEqualOpcode;
    use openvm_rv32im_circuit::{adapters::RV_B_TYPE_IMM_BITS, BranchEqualCoreAir};
    use openvm_rv32im_transpiler::BranchEqualOpcode;
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use test_case::test_case;

    use super::*;
    use crate::{extensions::native::write_native_or_imm, testing::GpuChipTestBuilder};

    const ABS_MAX_IMM: i32 = 1 << (RV_B_TYPE_IMM_BITS - 1);
    const MAX_INS_CAPACITY: usize = 128;
    type DenseChip<F> =
        NewVmChipWrapper<F, NativeBranchEqAir, NativeBranchEqStep, DenseRecordArena>;
    type SparseChip<F> =
        NewVmChipWrapper<F, NativeBranchEqAir, NativeBranchEqStep, MatrixRecordArena<F>>;

    fn create_test_dense_chip(tester: &GpuChipTestBuilder) -> DenseChip<F> {
        let mut chip = DenseChip::<F>::new(
            NativeBranchEqAir::new(
                BranchNativeAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                BranchEqualCoreAir::new(NativeBranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
            ),
            NativeBranchEqStep::new(
                BranchNativeAdapterStep::new(),
                NativeBranchEqualOpcode::CLASS_OFFSET,
                DEFAULT_PC_STEP,
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_test_sparse_chip(tester: &GpuChipTestBuilder) -> SparseChip<F> {
        let mut chip = SparseChip::<F>::new(
            NativeBranchEqAir::new(
                BranchNativeAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                BranchEqualCoreAir::new(NativeBranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
            ),
            NativeBranchEqStep::new(
                BranchNativeAdapterStep::new(),
                NativeBranchEqualOpcode::CLASS_OFFSET,
                DEFAULT_PC_STEP,
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    #[allow(clippy::too_many_arguments)]
    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        chip: &mut DenseChip<F>,
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
            chip,
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
        let mut tester = GpuChipTestBuilder::default().with_variable_range_checker();

        // CPU execution

        let mut dense_chip = create_test_dense_chip(&tester);

        let opcode = NativeBranchEqualOpcode(opcode);
        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut dense_chip, &mut rng, opcode);
        }

        let mut sparse_chip = create_test_sparse_chip(&tester);

        type Record<'a> = (
            &'a mut BranchNativeAdapterRecord<F>,
            &'a mut NativeBranchEqualCoreRecord<F>,
        );

        dense_chip
            .arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut sparse_chip.arena,
                EmptyAdapterCoreLayout::<F, BranchNativeAdapterStep>::new(),
            );

        // GPU tracegen
        let gpu_chip =
            NativeBranchEqChipGpu::new(dense_chip.air, tester.range_checker(), dense_chip.arena);

        // `gpu_chip` does GPU tracegen, `sparse_chip` does CPU tracegen. Must check that they are the same
        tester
            .build()
            .load_and_compare(gpu_chip, sparse_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
