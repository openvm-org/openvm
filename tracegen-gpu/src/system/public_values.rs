use std::{mem::size_of, sync::Arc};

use openvm_circuit::{
    arch::DenseRecordArena,
    system::{
        native_adapter::{NativeAdapterCols, NativeAdapterRecord},
        public_values::PublicValuesRecord,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::encoder::Encoder;
use openvm_stark_backend::{
    prover::{hal::MatrixDimensions, types::AirProvingContext},
    Chip,
};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prover_backend::GpuBackend, types::F,
};

use crate::{primitives::var_range::VariableRangeCheckerChipGPU, system::cuda};

#[repr(C)]
struct FullPublicValuesRecord {
    #[allow(unused)]
    adapter: NativeAdapterRecord<F, 2, 0>,
    #[allow(unused)]
    core: PublicValuesRecord<F>,
}

pub struct PublicValuesChipGPU {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub public_values: Vec<F>,
    pub num_custom_pvs: usize,
    pub max_degree: u32,
    // needed to compute the width of the trace
    encoder: Encoder,
    pub timestamp_max_bits: u32,
}

impl PublicValuesChipGPU {
    pub fn new(
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        num_custom_pvs: usize,
        max_degree: u32,
        timestamp_max_bits: u32,
    ) -> Self {
        Self {
            range_checker,
            public_values: Vec::new(),
            num_custom_pvs,
            max_degree,
            encoder: Encoder::new(num_custom_pvs, max_degree, true),
            timestamp_max_bits,
        }
    }
}

impl PublicValuesChipGPU {
    pub fn trace_height(arena: &DenseRecordArena) -> usize {
        let record_size = size_of::<FullPublicValuesRecord>();
        let records_len = arena.allocated().len();
        assert_eq!(records_len % record_size, 0);
        records_len / record_size
    }

    pub fn trace_width(&self) -> usize {
        NativeAdapterCols::<u8, 2, 0>::width() + 3 + self.encoder.width()
    }
}

impl Chip<DenseRecordArena, GpuBackend> for PublicValuesChipGPU {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let num_records = Self::trace_height(&arena);
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        unsafe {
            cuda::public_values::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &arena.allocated().to_device().unwrap(),
                num_records,
                &self.range_checker.count,
                self.timestamp_max_bits,
                self.num_custom_pvs,
                self.max_degree,
            )
            .expect("Failed to generate trace");
        }
        AirProvingContext::simple(trace, self.public_values.clone())
    }
}

#[cfg(test)]
mod tests {
    use openvm_circuit::{
        arch::{
            testing::{memory::gen_pointer, RANGE_CHECKER_BUS},
            EmptyAdapterCoreLayout, MemoryConfig, SystemConfig, VmChipWrapper,
        },
        system::{
            native_adapter::{NativeAdapterAir, NativeAdapterRecord, NativeAdapterStep},
            public_values::{
                PublicValuesAir, PublicValuesChip, PublicValuesCoreAir, PublicValuesRecord,
                PublicValuesStep,
            },
        },
    };
    use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
    use openvm_instructions::{
        instruction::Instruction, riscv::RV32_IMM_AS, LocalOpcode, PublishOpcode, NATIVE_AS,
    };
    use openvm_stark_sdk::utils::create_seeded_rng;
    use p3_field::{FieldAlgebra, PrimeField32};
    use rand::Rng;
    use stark_backend_gpu::types::F;

    use crate::{
        system::public_values::PublicValuesChipGPU,
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
    };

    type Harness = GpuTestChipHarness<
        F,
        PublicValuesStep<F>,
        PublicValuesAir,
        PublicValuesChipGPU,
        PublicValuesChip<F>,
    >;

    fn create_test_harness(
        tester: &GpuChipTestBuilder,
        mem_config: &MemoryConfig,
        system_config: &SystemConfig,
    ) -> Harness {
        let num_custom_pvs = system_config.num_public_values;
        let max_degree = system_config.max_constraint_degree as u32 - 1;
        let timestamp_max_bits = mem_config.clk_max_bits;

        let air = PublicValuesAir::new(
            NativeAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
            PublicValuesCoreAir::new(num_custom_pvs, max_degree),
        );

        let executor = PublicValuesStep::new(
            NativeAdapterStep::<F, 2, 0>::default(),
            num_custom_pvs,
            max_degree,
        );

        let cpu_chip = VmChipWrapper::new(
            PublicValuesStep::new(
                NativeAdapterStep::<F, 2, 0>::default(),
                num_custom_pvs,
                max_degree,
            ),
            tester.dummy_memory_helper(),
        );
        let gpu_chip = PublicValuesChipGPU::new(
            tester.range_checker(),
            num_custom_pvs,
            max_degree,
            timestamp_max_bits as u32,
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, num_custom_pvs)
    }

    #[test]
    fn test_public_values_tracegen() {
        let mut rng = create_seeded_rng();
        let system_config = SystemConfig::default();
        let mem_config = MemoryConfig::default();
        let bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, mem_config.decomp);
        let mut tester = GpuChipTestBuilder::volatile(mem_config.clone(), bus);

        let mut harness = create_test_harness(&tester, &mem_config, &system_config);
        let mut public_values = vec![];
        for idx in 0..system_config.num_public_values {
            let (b, e) = if rng.gen_bool(0.5) {
                let val = F::from_canonical_u32(rng.gen_range(0..F::ORDER_U32));
                public_values.push(val);
                (val, F::from_canonical_u32(RV32_IMM_AS))
            } else {
                let ptr = gen_pointer(&mut rng, 4);
                let val = F::from_canonical_u32(rng.gen_range(0..F::ORDER_U32));
                public_values.push(val);
                tester.write_cell(NATIVE_AS as usize, ptr, val);
                (
                    F::from_canonical_u32(ptr as u32),
                    F::from_canonical_u32(NATIVE_AS),
                )
            };

            let (c, f) = if rng.gen_bool(0.5) {
                (
                    F::from_canonical_u32(idx as u32),
                    F::from_canonical_u32(RV32_IMM_AS),
                )
            } else {
                let ptr = gen_pointer(&mut rng, 4);
                let val = F::from_canonical_u32(idx as u32);
                tester.write_cell(NATIVE_AS as usize, ptr, val);
                (
                    F::from_canonical_u32(ptr as u32),
                    F::from_canonical_u32(NATIVE_AS),
                )
            };

            let instruction = Instruction {
                opcode: PublishOpcode::PUBLISH.global_opcode(),
                a: F::ZERO,
                b,
                c,
                d: F::ZERO,
                e,
                f,
                g: F::ZERO,
            };
            tester.execute(
                &mut harness.executor,
                &mut harness.dense_arena,
                &instruction,
            );
        }
        harness.gpu_chip.public_values = public_values;

        type Record<'a> = (
            &'a mut NativeAdapterRecord<F, 2, 0>,
            &'a mut PublicValuesRecord<F>,
        );
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, NativeAdapterStep<F, 2, 0>>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .expect("Verification failed");
    }
}
