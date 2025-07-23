use std::{mem::size_of, sync::Arc};

use openvm_circuit::{
    arch::DenseRecordArena,
    system::{
        native_adapter::NativeAdapterRecord,
        public_values::{PublicValuesAir, PublicValuesRecord},
    },
    utils::next_power_of_two_or_zero,
};
use openvm_stark_backend::{
    prover::{hal::MatrixDimensions, types::AirProvingContext},
    Chip,
};
use p3_air::BaseAir;
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
    pub air: PublicValuesAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub public_values: Vec<F>,
    pub num_custom_pvs: usize,
    pub max_degree: u32,
    pub timestamp_max_bits: u32,
}

impl PublicValuesChipGPU {
    pub fn new(
        air: PublicValuesAir,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        num_custom_pvs: usize,
        max_degree: u32,
        timestamp_max_bits: u32,
    ) -> Self {
        Self {
            air,
            range_checker,
            public_values: Vec::new(),
            num_custom_pvs,
            max_degree,
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
        BaseAir::<F>::width(&self.air)
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
            testing::{memory::gen_pointer, TestChipHarness, RANGE_CHECKER_BUS},
            Arena, DenseRecordArena, EmptyAdapterCoreLayout, MatrixRecordArena, MemoryConfig,
            SystemConfig, VmChipWrapper,
        },
        system::{
            native_adapter::{NativeAdapterAir, NativeAdapterRecord, NativeAdapterStep},
            public_values::{
                PublicValuesAir, PublicValuesCoreAir, PublicValuesRecord, PublicValuesStep,
            },
        },
        utils::next_power_of_two_or_zero,
    };
    use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
    use openvm_instructions::{
        instruction::Instruction, riscv::RV32_IMM_AS, LocalOpcode, PublishOpcode, NATIVE_AS,
    };
    use openvm_stark_sdk::utils::create_seeded_rng;
    use p3_air::BaseAir;
    use p3_field::{FieldAlgebra, PrimeField32};
    use rand::Rng;
    use stark_backend_gpu::types::F;

    use crate::{
        system::public_values::PublicValuesChipGPU,
        testing::{dummy_memory_helper, GpuChipTestBuilder},
    };

    #[test]
    #[should_panic(expected = "LogUp multiset equality check failed.")]
    fn test_public_values_tracegen() {
        let system_config = SystemConfig::default();
        let mem_config = MemoryConfig::default();
        let bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, mem_config.decomp);
        let num_custom_pvs = system_config.num_public_values;
        let max_degree = system_config.max_constraint_degree as u32 - 1;
        let timestamp_max_bits = mem_config.clk_max_bits;

        let mut tester = GpuChipTestBuilder::volatile(mem_config, bus);
        let mut rng = create_seeded_rng();

        let executor = PublicValuesStep::new(
            NativeAdapterStep::<F, 2, 0>::default(),
            num_custom_pvs,
            max_degree,
        );
        let air = PublicValuesAir::new(
            NativeAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
            PublicValuesCoreAir::new(num_custom_pvs, max_degree),
        );
        let chip = PublicValuesChipGPU::new(
            air.clone(),
            tester.range_checker(),
            num_custom_pvs,
            max_degree,
            timestamp_max_bits as u32,
        );
        let mut harness = TestChipHarness::with_capacity(executor, air, chip, num_custom_pvs);

        let mut public_values = vec![];
        for idx in 0..num_custom_pvs {
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
            tester.execute_harness::<_, _, _, DenseRecordArena>(&mut harness, &instruction);
        }
        harness.chip.public_values = public_values;

        type Record<'a> = (
            &'a mut NativeAdapterRecord<F, 2, 0>,
            &'a mut PublicValuesRecord<F>,
        );
        let cpu_chip = VmChipWrapper::new(
            PublicValuesStep::new(
                NativeAdapterStep::<F, 2, 0>::default(),
                num_custom_pvs,
                max_degree,
            ),
            dummy_memory_helper(bus, timestamp_max_bits),
        );
        let mut cpu_arena = MatrixRecordArena::<F>::with_capacity(
            next_power_of_two_or_zero(num_custom_pvs),
            <PublicValuesAir as BaseAir<F>>::width(&harness.air),
        );
        harness
            .arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut cpu_arena,
                EmptyAdapterCoreLayout::<F, NativeAdapterStep<F, 2, 0>>::new(),
            );

        tester
            .build()
            .load_and_compare(
                harness.air,
                harness.chip,
                harness.arena,
                cpu_chip,
                cpu_arena,
            )
            .finalize()
            .simple_test()
            .expect("Verification failed");
    }
}
