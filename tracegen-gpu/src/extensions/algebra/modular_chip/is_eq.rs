use std::{mem::size_of, sync::Arc};

use openvm_algebra_circuit::modular_chip::{ModularIsEqualAir, ModularIsEqualRecord};
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_rv32_adapters::Rv32IsEqualModAdapterRecord;
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend, types::SC,
};

use crate::{
    extensions::algebra::cuda::is_eq_cuda::tracegen,
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
    DeviceChip,
};

pub struct ModularIsEqualChipGpu<
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
> {
    air: ModularIsEqualAir<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    arena: DenseRecordArena,
}

impl<const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize>
    ModularIsEqualChipGpu<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>
{
    pub fn new(
        air: ModularIsEqualAir<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
        arena: DenseRecordArena,
    ) -> Self {
        Self {
            air,
            range_checker,
            bitwise_lookup,
            arena,
        }
    }
}

impl<const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize> ChipUsageGetter
    for ModularIsEqualChipGpu<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>
{
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        let record_size = size_of::<(
            Rv32IsEqualModAdapterRecord<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
            ModularIsEqualRecord<TOTAL_LIMBS>,
        )>();
        let records_len = self.arena.allocated().len();
        assert_eq!(records_len % record_size, 0);
        records_len / record_size
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl<const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize>
    DeviceChip<SC, GpuBackend> for ModularIsEqualChipGpu<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records = self.arena.allocated().to_device().unwrap();
        let trace_height = next_power_of_two_or_zero(self.current_trace_height());
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());

        let mut mod_bytes = [0u8; TOTAL_LIMBS];
        for (i, &limb) in self.air.core.modulus_limbs.iter().enumerate() {
            mod_bytes[i] = limb as u8;
        }
        let d_modulus = mod_bytes.as_slice().to_device().unwrap();

        unsafe {
            tracegen(
                trace.buffer(),
                trace_height,
                &d_records,
                &d_modulus,
                TOTAL_LIMBS,
                NUM_LANES,
                LANE_SIZE,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
            )
            .unwrap();
        }

        trace
    }
}

#[cfg(test)]
mod tests {
    use openvm_algebra_circuit::modular_chip::{
        ModularIsEqualAir, ModularIsEqualCoreAir, VmModularIsEqualStep,
    };
    use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
    use openvm_circuit::arch::{
        testing::{memory::gen_pointer, BITWISE_OP_LOOKUP_BUS},
        DenseRecordArena, EmptyAdapterCoreLayout, MatrixRecordArena, NewVmChipWrapper,
        VmAirWrapper,
    };
    use openvm_circuit_primitives::{
        bigint::utils::{big_uint_to_limbs, secp256k1_coord_prime},
        bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    };
    use openvm_instructions::{instruction::Instruction, riscv::RV32_CELL_BITS};
    use openvm_rv32_adapters::{
        Rv32IsEqualModAdapterAir, Rv32IsEqualModAdapterRecord, Rv32IsEqualModeAdapterStep,
    };
    use openvm_rv32im_circuit::adapters::RV32_REGISTER_NUM_LIMBS;
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};

    use super::*;
    use crate::testing::GpuChipTestBuilder;

    const NUM_LANES: usize = 1;
    const LANE_SIZE: usize = 32;
    const TOTAL_LIMBS: usize = 32;
    const MAX_INS_CAPACITY: usize = 512;
    const OPCODE_OFFSET: usize = 17;

    type DenseChip = NewVmChipWrapper<
        F,
        ModularIsEqualAir<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        VmModularIsEqualStep<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        DenseRecordArena,
    >;

    type SparseChip = NewVmChipWrapper<
        F,
        ModularIsEqualAir<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        VmModularIsEqualStep<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        MatrixRecordArena<F>,
    >;

    fn create_dense_chip(
        tester: &GpuChipTestBuilder,
        modulus: &num_bigint::BigUint,
        modulus_limbs: [u8; TOTAL_LIMBS],
        bitwise_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> DenseChip {
        let mut chip = NewVmChipWrapper::<_, _, _, DenseRecordArena>::new(
            VmAirWrapper::new(
                Rv32IsEqualModAdapterAir::<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>::new(
                    tester.execution_bridge(),
                    tester.memory_bridge(),
                    bitwise_chip.bus(),
                    tester.address_bits(),
                ),
                ModularIsEqualCoreAir::<TOTAL_LIMBS, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>::new(
                    modulus.clone(),
                    bitwise_chip.bus(),
                    OPCODE_OFFSET,
                ),
            ),
            VmModularIsEqualStep::new(
                Rv32IsEqualModeAdapterStep::<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>::new(
                    tester.address_bits(),
                    bitwise_chip.clone(),
                ),
                modulus_limbs,
                OPCODE_OFFSET,
                bitwise_chip.clone(),
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_sparse_chip(
        tester: &GpuChipTestBuilder,
        modulus: &num_bigint::BigUint,
        modulus_limbs: [u8; TOTAL_LIMBS],
        bitwise_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> SparseChip {
        let mut chip = NewVmChipWrapper::<_, _, _, MatrixRecordArena<F>>::new(
            VmAirWrapper::new(
                Rv32IsEqualModAdapterAir::<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>::new(
                    tester.execution_bridge(),
                    tester.memory_bridge(),
                    bitwise_chip.bus(),
                    tester.address_bits(),
                ),
                ModularIsEqualCoreAir::<TOTAL_LIMBS, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>::new(
                    modulus.clone(),
                    bitwise_chip.bus(),
                    OPCODE_OFFSET,
                ),
            ),
            VmModularIsEqualStep::new(
                Rv32IsEqualModeAdapterStep::<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>::new(
                    tester.address_bits(),
                    bitwise_chip.clone(),
                ),
                modulus_limbs,
                OPCODE_OFFSET,
                bitwise_chip.clone(),
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        chip: &mut DenseChip,
        rng: &mut StdRng,
        modulus_limbs: [F; TOTAL_LIMBS],
        is_setup: bool,
        b: Option<[F; TOTAL_LIMBS]>,
        c: Option<[F; TOTAL_LIMBS]>,
    ) {
        let a_ptr: u32 = gen_pointer(rng, 4) as u32;
        let b_ptr: u32 = gen_pointer(rng, 4) as u32;
        let c_ptr: u32 = gen_pointer(rng, 4) as u32;

        let (b_data, c_data, opcode) = if is_setup {
            (
                modulus_limbs,
                [F::ZERO; TOTAL_LIMBS],
                Rv32ModularArithmeticOpcode::SETUP_ISEQ,
            )
        } else {
            let b_data =
                b.unwrap_or_else(|| std::array::from_fn(|_| F::from_canonical_u8(rng.gen::<u8>())));
            let c_data = c.unwrap_or_else(|| {
                if rng.gen_bool(0.5) {
                    b_data
                } else {
                    std::array::from_fn(|_| F::from_canonical_u8(rng.gen::<u8>()))
                }
            });
            (b_data, c_data, Rv32ModularArithmeticOpcode::IS_EQ)
        };

        tester.write(2, b_ptr as usize, b_data);
        tester.write(2, c_ptr as usize, c_data);
        tester.write::<4>(
            1,
            b_ptr as usize,
            b_ptr.to_le_bytes().map(F::from_canonical_u8),
        );
        tester.write::<4>(
            1,
            c_ptr as usize,
            c_ptr.to_le_bytes().map(F::from_canonical_u8),
        );

        let instruction = Instruction::new(
            openvm_instructions::VmOpcode::from_usize(opcode as usize + OPCODE_OFFSET),
            F::from_canonical_u32(a_ptr),
            F::from_canonical_u32(b_ptr),
            F::from_canonical_u32(c_ptr),
            F::from_canonical_u32(1),
            F::from_canonical_u32(2),
            F::ZERO,
            F::ZERO,
        );

        tester.execute(chip, &instruction);
    }

    #[test]
    fn test_modular_is_equal_tracegen() {
        let mut rng = create_seeded_rng();
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let mut tester = GpuChipTestBuilder::default()
            .with_variable_range_checker()
            .with_bitwise_op_lookup(bitwise_bus);

        let modulus = secp256k1_coord_prime();
        let limbs = big_uint_to_limbs(&modulus, 8);
        let modulus_limbs: [u8; TOTAL_LIMBS] = {
            let mut arr = [0u8; TOTAL_LIMBS];
            for (i, &val) in limbs.iter().enumerate() {
                if i < TOTAL_LIMBS {
                    arr[i] = val as u8;
                }
            }
            arr
        };
        let modulus_limbs_f = modulus_limbs.map(F::from_canonical_u8);

        let cpu_bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
        let mut dense_cpu =
            create_dense_chip(&tester, &modulus, modulus_limbs, cpu_bitwise_chip.clone());

        for i in 0..100 {
            set_and_execute(
                &mut tester,
                &mut dense_cpu,
                &mut rng,
                modulus_limbs_f,
                i == 0,
                None,
                None,
            );
        }

        let mut b = modulus_limbs_f;
        b[0] -= F::ONE;
        set_and_execute(
            &mut tester,
            &mut dense_cpu,
            &mut rng,
            modulus_limbs_f,
            false,
            Some(b),
            Some(b),
        );

        let mut sparse_cpu =
            create_sparse_chip(&tester, &modulus, modulus_limbs, cpu_bitwise_chip.clone());

        type Record<'a> = (
            &'a mut Rv32IsEqualModAdapterRecord<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
            &'a mut ModularIsEqualRecord<TOTAL_LIMBS>,
        );
        dense_cpu
            .arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut sparse_cpu.arena,
                EmptyAdapterCoreLayout::<
                    F,
                    Rv32IsEqualModeAdapterStep<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
                >::new(),
            );

        let gpu_chip = ModularIsEqualChipGpu::<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>::new(
            dense_cpu.air.clone(),
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            dense_cpu.arena,
        );

        tester
            .build()
            .load_and_compare(gpu_chip, sparse_cpu)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
