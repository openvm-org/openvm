use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_native_circuit::{
    adapters::{NativeVectorizedAdapterCols, NativeVectorizedAdapterRecord},
    FieldExtensionCoreCols, FieldExtensionRecord, EXT_DEG,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prover_backend::GpuBackend, types::F,
};

use crate::{
    extensions::native::field_extension_cuda, get_empty_air_proving_ctx,
    primitives::var_range::VariableRangeCheckerChipGPU,
};

#[derive(new)]
pub struct FieldExtensionChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for FieldExtensionChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            NativeVectorizedAdapterRecord<F, EXT_DEG>,
            FieldExtensionRecord<F>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        assert_eq!(records.len() % RECORD_SIZE, 0);

        let height = records.len() / RECORD_SIZE;
        let padded_height = next_power_of_two_or_zero(height);
        let trace_width = NativeVectorizedAdapterCols::<F, EXT_DEG>::width()
            + FieldExtensionCoreCols::<F>::width();
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, trace_width);

        let d_records = records.to_device().unwrap();

        unsafe {
            field_extension_cuda::tracegen(
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
mod tests {
    use openvm_circuit::arch::{testing::memory::gen_pointer, EmptyAdapterCoreLayout};
    use openvm_instructions::{instruction::Instruction, LocalOpcode};
    use openvm_native_circuit::{
        adapters::{
            NativeVectorizedAdapterAir, NativeVectorizedAdapterExecutor,
            NativeVectorizedAdapterFiller,
        },
        FieldExtensionAir, FieldExtensionChip, FieldExtensionCoreAir, FieldExtensionCoreFiller,
        FieldExtensionExecutor,
    };
    use openvm_native_compiler::{conversion::AS, FieldExtensionOpcode};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::rngs::StdRng;
    use test_case::test_case;

    use super::*;
    use crate::{
        extensions::native::write_native_array,
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
    };

    const MAX_INS_CAPACITY: usize = 128;

    fn create_test_harness(
        tester: &GpuChipTestBuilder,
    ) -> GpuTestChipHarness<
        F,
        FieldExtensionExecutor,
        FieldExtensionAir,
        FieldExtensionChipGpu,
        FieldExtensionChip<F>,
    > {
        let adapter_air =
            NativeVectorizedAdapterAir::new(tester.execution_bridge(), tester.memory_bridge());
        let core_air = FieldExtensionCoreAir::new();
        let air = FieldExtensionAir::new(adapter_air, core_air);

        let adapter_step = NativeVectorizedAdapterExecutor::new();
        let executor = FieldExtensionExecutor::new(adapter_step);

        let core_filler = FieldExtensionCoreFiller::new(NativeVectorizedAdapterFiller);

        let cpu_chip = FieldExtensionChip::new(core_filler, tester.dummy_memory_helper());
        let gpu_chip =
            FieldExtensionChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        harness: &mut GpuTestChipHarness<
            F,
            FieldExtensionExecutor,
            FieldExtensionAir,
            FieldExtensionChipGpu,
            FieldExtensionChip<F>,
        >,
        rng: &mut StdRng,
        opcode: FieldExtensionOpcode,
    ) {
        let (_, y_ptr) = write_native_array::<EXT_DEG>(tester, rng, None);
        let (_, z_ptr) = write_native_array::<EXT_DEG>(tester, rng, None);

        let x_ptr = gen_pointer(rng, EXT_DEG);

        tester.execute(
            &mut harness.executor,
            &mut harness.dense_arena,
            &Instruction::from_usize(
                opcode.global_opcode(),
                [
                    x_ptr,
                    y_ptr,
                    z_ptr,
                    AS::Native as usize,
                    AS::Native as usize,
                ],
            ),
        );
    }

    #[test_case(FieldExtensionOpcode::FE4ADD, 100)]
    #[test_case(FieldExtensionOpcode::FE4SUB, 100)]
    #[test_case(FieldExtensionOpcode::BBE4MUL, 100)]
    #[test_case(FieldExtensionOpcode::BBE4DIV, 100)]
    fn rand_field_extension_tracegen_test(opcode: FieldExtensionOpcode, num_ops: usize) {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default();

        let mut harness = create_test_harness(&tester);

        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut harness, &mut rng, opcode);
        }

        type Record<'a> = (
            &'a mut NativeVectorizedAdapterRecord<F, EXT_DEG>,
            &'a mut FieldExtensionRecord<F>,
        );

        harness
            .dense_arena
            .get_record_seeker::<Record<'_>, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, NativeVectorizedAdapterExecutor<EXT_DEG>>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
