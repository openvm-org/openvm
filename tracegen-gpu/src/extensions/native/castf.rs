use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_native_circuit::{adapters::ConvertAdapterRecord, CastFAir, CastFCoreRecord};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::copy::MemCopyH2D,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::{
    extensions::native::castf_cuda, primitives::var_range::VariableRangeCheckerChipGPU, DeviceChip,
};

#[derive(new)]
pub struct CastFChipGpu {
    pub air: CastFAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub arena: DenseRecordArena,
}

impl ChipUsageGetter for CastFChipGpu {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(ConvertAdapterRecord<F, 1, 4>, CastFCoreRecord)>();
        let records_len = self.arena.allocated().len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for CastFChipGpu {
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
            castf_cuda::tracegen(
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
    use openvm_circuit::arch::{
        testing::memory::gen_pointer, EmptyAdapterCoreLayout, MatrixRecordArena, NewVmChipWrapper,
    };
    use openvm_instructions::{
        instruction::Instruction,
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_NUM_LIMBS},
        LocalOpcode,
    };
    use openvm_native_circuit::{
        adapters::{ConvertAdapterAir, ConvertAdapterStep},
        CastFCoreAir, CastFStep,
    };
    use openvm_native_compiler::{conversion::AS, CastfOpcode};
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};

    use super::*;
    use crate::{extensions::native::write_native_array, testing::GpuChipTestBuilder};

    const CASTF_MAX_BITS: usize = 30;
    const MAX_INS_CAPACITY: usize = 128;
    type DenseChip<F> = NewVmChipWrapper<F, CastFAir, CastFStep, DenseRecordArena>;
    type SparseChip<F> = NewVmChipWrapper<F, CastFAir, CastFStep, MatrixRecordArena<F>>;

    fn create_test_dense_chip(tester: &GpuChipTestBuilder) -> DenseChip<F> {
        let mut chip = DenseChip::<F>::new(
            CastFAir::new(
                ConvertAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                CastFCoreAir::new(tester.cpu_range_checker().bus()),
            ),
            CastFStep::new(
                ConvertAdapterStep::<1, 4>::new(),
                tester.cpu_range_checker(),
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_test_sparse_chip(tester: &GpuChipTestBuilder) -> SparseChip<F> {
        let mut chip = SparseChip::<F>::new(
            CastFAir::new(
                ConvertAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                CastFCoreAir::new(tester.cpu_range_checker().bus()),
            ),
            CastFStep::new(
                ConvertAdapterStep::<1, 4>::new(),
                tester.cpu_range_checker(),
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
        b: Option<F>,
    ) {
        let b_val = b.unwrap_or(F::from_canonical_u32(rng.gen_range(0..1 << CASTF_MAX_BITS)));
        let b_ptr = write_native_array(tester, rng, Some([b_val])).1;

        let a = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        tester.execute(
            chip,
            &Instruction::from_usize(
                CastfOpcode::CASTF.global_opcode(),
                [a, b_ptr, 0, RV32_MEMORY_AS as usize, AS::Native as usize],
            ),
        );
    }

    #[test]
    fn test_castf_tracegen() {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default().with_variable_range_checker();

        // CPU execution
        let mut dense_chip = create_test_dense_chip(&tester);

        let num_ops = 100;
        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut dense_chip, &mut rng, None);
        }

        set_and_execute(&mut tester, &mut dense_chip, &mut rng, Some(F::ZERO));

        let mut sparse_chip = create_test_sparse_chip(&tester);

        type Record<'a> = (
            &'a mut ConvertAdapterRecord<F, 1, 4>,
            &'a mut CastFCoreRecord,
        );

        dense_chip
            .arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut sparse_chip.arena,
                EmptyAdapterCoreLayout::<F, ConvertAdapterStep<1, 4>>::new(),
            );

        // GPU tracegen
        let gpu_chip = CastFChipGpu::new(dense_chip.air, tester.range_checker(), dense_chip.arena);

        // `gpu_chip` does GPU tracegen, `sparse_chip` does CPU tracegen. Must check that they are the same
        tester
            .build()
            .load_and_compare(gpu_chip, sparse_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
