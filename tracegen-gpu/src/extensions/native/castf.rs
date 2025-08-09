use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_native_circuit::{
    adapters::{ConvertAdapterCols, ConvertAdapterRecord},
    CastFCoreCols, CastFCoreRecord,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prover_backend::GpuBackend, types::F,
};

use crate::{
    extensions::native::castf_cuda, get_empty_air_proving_ctx,
    primitives::var_range::VariableRangeCheckerChipGPU,
};

#[derive(new)]
pub struct CastFChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for CastFChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(ConvertAdapterRecord<F, 1, 4>, CastFCoreRecord)>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        assert_eq!(records.len() % RECORD_SIZE, 0);

        let height = records.len() / RECORD_SIZE;
        let padded_height = next_power_of_two_or_zero(height);
        let trace_width = ConvertAdapterCols::<F, 1, 4>::width() + CastFCoreCols::<F>::width();
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, trace_width);

        let d_records = records.to_device().unwrap();

        unsafe {
            castf_cuda::tracegen(
                trace.buffer(),
                padded_height as u32,
                trace_width as u32,
                &d_records,
                height as u32,
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
    use openvm_circuit::arch::{testing::memory::gen_pointer, EmptyAdapterCoreLayout};
    use openvm_instructions::{
        instruction::Instruction,
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_NUM_LIMBS},
        LocalOpcode,
    };
    use openvm_native_circuit::{
        adapters::{ConvertAdapterAir, ConvertAdapterExecutor, ConvertAdapterFiller},
        CastFAir, CastFChip, CastFCoreAir, CastFCoreFiller, CastFExecutor,
    };
    use openvm_native_compiler::{conversion::AS, CastfOpcode};
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use test_case::test_case;

    use super::*;
    use crate::{
        extensions::native::write_native_array,
        testing::{
            default_var_range_checker_bus, dummy_range_checker, GpuChipTestBuilder,
            GpuTestChipHarness,
        },
    };

    const CASTF_MAX_BITS: usize = 30;
    const MAX_INS_CAPACITY: usize = 128;

    fn create_test_harness(
        tester: &GpuChipTestBuilder,
    ) -> GpuTestChipHarness<F, CastFExecutor, CastFAir, CastFChipGpu, CastFChip<F>> {
        let range_bus = default_var_range_checker_bus();
        let adapter_air = ConvertAdapterAir::new(tester.execution_bridge(), tester.memory_bridge());
        let core_air = CastFCoreAir::new(range_bus);
        let air = CastFAir::new(adapter_air, core_air);

        let adapter_step = ConvertAdapterExecutor::<1, 4>::new();
        let executor = CastFExecutor::new(adapter_step);

        let core_filler =
            CastFCoreFiller::new(ConvertAdapterFiller, dummy_range_checker(range_bus));

        let cpu_chip = CastFChip::new(core_filler, tester.dummy_memory_helper());
        let gpu_chip = CastFChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    #[allow(clippy::too_many_arguments)]
    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        harness: &mut GpuTestChipHarness<F, CastFExecutor, CastFAir, CastFChipGpu, CastFChip<F>>,
        rng: &mut StdRng,
        b: Option<F>,
    ) {
        let b_val = b.unwrap_or(F::from_canonical_u32(rng.gen_range(0..1 << CASTF_MAX_BITS)));
        let b_ptr = write_native_array(tester, rng, Some([b_val])).1;

        let a = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        tester.execute(
            &mut harness.executor,
            &mut harness.dense_arena,
            &Instruction::from_usize(
                CastfOpcode::CASTF.global_opcode(),
                [a, b_ptr, 0, RV32_MEMORY_AS as usize, AS::Native as usize],
            ),
        );
    }

    #[test_case(100)]
    fn test_castf_tracegen(num_ops: usize) {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default();

        let mut harness = create_test_harness(&tester);

        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut harness, &mut rng, None);
        }

        set_and_execute(&mut tester, &mut harness, &mut rng, Some(F::ZERO));

        type Record<'a> = (
            &'a mut ConvertAdapterRecord<F, 1, 4>,
            &'a mut CastFCoreRecord,
        );

        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, ConvertAdapterExecutor<1, 4>>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
