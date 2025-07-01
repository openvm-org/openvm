use std::{mem::size_of, sync::Arc};

use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_native_circuit::{
    adapters::AluNativeAdapterRecord, FieldArithmeticAir, FieldArithmeticRecord,
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
    extensions::native::field_arithmetic_cuda, primitives::var_range::VariableRangeCheckerChipGPU,
    DeviceChip,
};

pub struct FieldArithmeticChipGpu<'a> {
    pub air: FieldArithmeticAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl<'a> FieldArithmeticChipGpu<'a> {
    pub fn new(
        air: FieldArithmeticAir,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        arena: Option<&'a DenseRecordArena>,
    ) -> Self {
        Self {
            air,
            range_checker,
            arena,
        }
    }
}

impl ChipUsageGetter for FieldArithmeticChipGpu<'_> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize =
            size_of::<(AluNativeAdapterRecord<F>, FieldArithmeticRecord<F>)>();
        let buf = &self.arena.unwrap().allocated();
        buf.len() / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        use openvm_stark_backend::p3_air::BaseAir;
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for FieldArithmeticChipGpu<'_> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let buf = &self.arena.unwrap().allocated();
        let d_records: DeviceBuffer<u8> = buf.to_device().unwrap();
        let height = next_power_of_two_or_zero(self.current_trace_height());
        let trace = DeviceMatrix::<F>::with_capacity(height, self.trace_width());
        unsafe {
            field_arithmetic_cuda::tracegen(
                trace.buffer(),
                height,
                self.trace_width(),
                &d_records,
                self.range_checker.count.as_ptr() as *const u32,
                self.range_checker.count.len(),
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
        adapters::{AluNativeAdapterAir, AluNativeAdapterStep},
        FieldArithmeticCoreAir, FieldArithmeticStep,
    };
    use openvm_native_compiler::{conversion::AS, FieldArithmeticOpcode};
    use openvm_stark_backend::{
        p3_field::{Field, FieldAlgebra},
        verifier::VerificationError,
    };
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::Rng;
    use test_case::test_case;

    use super::*;
    use crate::{extensions::native::write_native_array, testing::GpuChipTestBuilder};

    const MAX_INS_CAPACITY: usize = 128;

    fn create_dense_native_chip(
        tester: &GpuChipTestBuilder,
    ) -> NewVmChipWrapper<F, FieldArithmeticAir, FieldArithmeticStep, DenseRecordArena> {
        let mut chip =
            NewVmChipWrapper::<F, FieldArithmeticAir, FieldArithmeticStep, DenseRecordArena>::new(
                VmAirWrapper::new(
                    AluNativeAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                    FieldArithmeticCoreAir::new(),
                ),
                FieldArithmeticStep::new(AluNativeAdapterStep::new()),
                tester.cpu_memory_helper(),
            );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_sparse_native_chip(
        tester: &GpuChipTestBuilder,
    ) -> NewVmChipWrapper<F, FieldArithmeticAir, FieldArithmeticStep, MatrixRecordArena<F>> {
        let mut chip = NewVmChipWrapper::<
            F,
            FieldArithmeticAir,
            FieldArithmeticStep,
            MatrixRecordArena<F>,
        >::new(
            VmAirWrapper::new(
                AluNativeAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                FieldArithmeticCoreAir::new(),
            ),
            FieldArithmeticStep::new(AluNativeAdapterStep::new()),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    #[test_case(FieldArithmeticOpcode::ADD)]
    #[test_case(FieldArithmeticOpcode::SUB)]
    #[test_case(FieldArithmeticOpcode::MUL)]
    #[test_case(FieldArithmeticOpcode::DIV)]
    fn rand_field_arithmetic_tracegen_test(opcode: FieldArithmeticOpcode) {
        let mut tester = GpuChipTestBuilder::default().with_variable_range_checker();
        let mut rng = create_seeded_rng();

        let mut dense_chip = create_dense_native_chip(&tester);
        let mut gpu_chip =
            FieldArithmeticChipGpu::new(dense_chip.air, tester.range_checker(), None);
        let mut cpu_chip = create_sparse_native_chip(&tester);

        for _ in 0..100 {
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
                &mut dense_chip,
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

        dense_chip
            .arena
            .get_record_seeker::<Record<'_>, _>()
            .transfer_to_matrix_arena(
                &mut cpu_chip.arena,
                EmptyAdapterCoreLayout::<F, AluNativeAdapterStep>::new(),
            );
        gpu_chip.arena = Some(&dense_chip.arena);

        tester
            .build()
            .load_and_compare(gpu_chip, cpu_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
