use std::{mem::size_of, sync::Arc};

use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_native_circuit::{
    adapters::NativeVectorizedAdapterRecord, FieldExtensionAir, FieldExtensionRecord, EXT_DEG,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer},
    prelude::F,
    prover_backend::GpuBackend,
    types::SC,
};

use crate::{
    extensions::native::field_extension_cuda, primitives::var_range::VariableRangeCheckerChipGPU,
    DeviceChip,
};

#[derive(derive_new::new)]
pub struct FieldExtensionChipGpu {
    pub air: FieldExtensionAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub arena: DenseRecordArena,
}

impl ChipUsageGetter for FieldExtensionChipGpu {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(
            NativeVectorizedAdapterRecord<F, EXT_DEG>,
            FieldExtensionRecord<F>,
        )>();
        let buf = self.arena.allocated();
        buf.len() / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        use openvm_stark_backend::p3_air::BaseAir;
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for FieldExtensionChipGpu {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records: DeviceBuffer<u8> = self.arena.allocated().to_device().unwrap();
        let height = self.current_trace_height();
        let padded_height = next_power_of_two_or_zero(height);
        let width = self.trace_width();
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, width);
        unsafe {
            field_extension_cuda::tracegen(
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
mod tests {
    use openvm_circuit::arch::{
        testing::memory::gen_pointer, DenseRecordArena, EmptyAdapterCoreLayout, MatrixRecordArena,
        NewVmChipWrapper, VmAirWrapper,
    };
    use openvm_instructions::{instruction::Instruction, LocalOpcode};
    use openvm_native_circuit::{
        adapters::{NativeVectorizedAdapterAir, NativeVectorizedAdapterStep},
        FieldExtensionCoreAir, FieldExtensionStep,
    };
    use openvm_native_compiler::{conversion::AS, FieldExtensionOpcode};
    use openvm_stark_backend::verifier::VerificationError;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::rngs::StdRng;
    use test_case::test_case;

    use super::*;
    use crate::{extensions::native::write_native_array, testing::GpuChipTestBuilder};

    const MAX_INS_CAPACITY: usize = 128;
    type DenseChip<F> =
        NewVmChipWrapper<F, FieldExtensionAir, FieldExtensionStep, DenseRecordArena>;
    type SparseChip<F> =
        NewVmChipWrapper<F, FieldExtensionAir, FieldExtensionStep, MatrixRecordArena<F>>;

    fn create_test_dense_chip(tester: &GpuChipTestBuilder) -> DenseChip<F> {
        let mut chip = DenseChip::new(
            VmAirWrapper::new(
                NativeVectorizedAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                FieldExtensionCoreAir::new(),
            ),
            FieldExtensionStep::new(NativeVectorizedAdapterStep::new()),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_test_sparse_chip(tester: &GpuChipTestBuilder) -> SparseChip<F> {
        let mut chip = SparseChip::new(
            VmAirWrapper::new(
                NativeVectorizedAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                FieldExtensionCoreAir::new(),
            ),
            FieldExtensionStep::new(NativeVectorizedAdapterStep::new()),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        chip: &mut DenseChip<F>,
        rng: &mut StdRng,
        opcode: FieldExtensionOpcode,
    ) {
        let (_, y_ptr) = write_native_array::<EXT_DEG>(tester, rng, None);
        let (_, z_ptr) = write_native_array::<EXT_DEG>(tester, rng, None);

        let x_ptr = gen_pointer(rng, EXT_DEG);

        tester.execute(
            chip,
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

    #[test_case(FieldExtensionOpcode::FE4ADD)]
    #[test_case(FieldExtensionOpcode::FE4SUB)]
    #[test_case(FieldExtensionOpcode::BBE4MUL)]
    #[test_case(FieldExtensionOpcode::BBE4DIV)]
    fn rand_field_extension_tracegen_test(opcode: FieldExtensionOpcode) {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default().with_variable_range_checker();

        // CPU execution
        let mut dense_chip = create_test_dense_chip(&tester);

        for _ in 0..100 {
            set_and_execute(&mut tester, &mut dense_chip, &mut rng, opcode);
        }

        let mut sparse_chip = create_test_sparse_chip(&tester);

        type Record<'a> = (
            &'a mut NativeVectorizedAdapterRecord<F, EXT_DEG>,
            &'a mut FieldExtensionRecord<F>,
        );

        dense_chip
            .arena
            .get_record_seeker::<Record<'_>, _>()
            .transfer_to_matrix_arena(
                &mut sparse_chip.arena,
                EmptyAdapterCoreLayout::<F, NativeVectorizedAdapterStep<EXT_DEG>>::new(),
            );

        // GPU tracegen
        let gpu_chip =
            FieldExtensionChipGpu::new(dense_chip.air, tester.range_checker(), dense_chip.arena);

        tester
            .build()
            .load_and_compare(gpu_chip, sparse_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
