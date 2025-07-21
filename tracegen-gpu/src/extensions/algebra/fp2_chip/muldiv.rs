use std::sync::Arc;

use derive_new::new;
use openvm_circuit::arch::{AdapterCoreLayout, DenseRecordArena, RecordSeeker, VmAirWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionMetadata};
use openvm_rv32_adapters::{
    Rv32VecHeapAdapterAir, Rv32VecHeapAdapterRecord, Rv32VecHeapAdapterStep,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend, types::SC,
};

use crate::{
    mod_builder::field_expression::{constants::LIMB_BITS, FieldExpressionChipGPU},
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
    DeviceChip,
};

type Fp2Air<const BLOCKS: usize, const BLOCK_SIZE: usize> = VmAirWrapper<
    Rv32VecHeapAdapterAir<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

#[derive(new)]
pub struct Fp2MulDivChipGpu<'a, const BLOCKS: usize, const BLOCK_SIZE: usize> {
    pub air: Fp2Air<BLOCKS, BLOCK_SIZE>,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<LIMB_BITS>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Fp2MulDivChipGpu<'_, BLOCKS, BLOCK_SIZE> {
    fn get_record_size(&self) -> usize {
        let total_input_limbs =
            self.air.core.expr.builder.num_input * self.air.core.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterStep<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        RecordSeeker::<
            DenseRecordArena,
            (
                &mut Rv32VecHeapAdapterRecord<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
                openvm_mod_circuit_builder::FieldExpressionCoreRecordMut<'_>,
            ),
            _,
        >::get_aligned_record_size(&layout)
    }
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> ChipUsageGetter
    for Fp2MulDivChipGpu<'_, BLOCKS, BLOCK_SIZE>
{
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        let record_size = self.get_record_size();
        let buf = &self.arena.unwrap().allocated();
        let total = buf.len();

        assert_eq!(total % record_size, 0);
        total / record_size
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> DeviceChip<SC, GpuBackend>
    for Fp2MulDivChipGpu<'_, BLOCKS, BLOCK_SIZE>
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let buf = &self.arena.unwrap().allocated();
        let d_records = buf.to_device().unwrap();

        let record_size = self.get_record_size();
        let num_records = buf.len() / record_size;

        let adapter_width =
            <Rv32VecHeapAdapterAir<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> as BaseAir<F>>::width(
                &self.air.adapter,
            );
        let core_chip = FieldExpressionChipGPU::new(
            self.air.core.clone(),
            d_records,
            num_records,
            record_size,
            adapter_width,
            BLOCKS,
            self.range_checker.clone(),
            self.bitwise_lookup.clone(),
        );

        core_chip.generate_field_trace()
    }
}

#[cfg(test)]
mod tests {
    use openvm_algebra_circuit::FieldExprVecHeapStep;
    use openvm_algebra_transpiler::Fp2Opcode;
    use openvm_circuit::arch::{
        testing::BITWISE_OP_LOOKUP_BUS, DenseRecordArena, MatrixRecordArena, NewVmChipWrapper,
    };
    use openvm_circuit_primitives::{
        bigint::utils::secp256k1_coord_prime,
        bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    };
    use openvm_instructions::{
        instruction::Instruction, riscv::RV32_CELL_BITS, LocalOpcode, VmOpcode,
    };
    use openvm_mod_circuit_builder::{test_utils::biguint_to_limbs, ExprBuilderConfig};
    use openvm_rv32_adapters::Rv32VecHeapAdapterRecord;
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
    use rand::Rng;
    use stark_backend_gpu::prelude::F;

    use super::*;
    use crate::testing::GpuChipTestBuilder;

    // Reuse BLOCKS = 2, BLOCK_SIZE = NUM_LIMBS for Fp2 (two field elements)
    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    const MAX_INS_CAPACITY: usize = 512;

    type Fp2Step<const BLOCKS: usize, const BLOCK_SIZE: usize> =
        FieldExprVecHeapStep<2, BLOCKS, BLOCK_SIZE>;

    // Helper to build dense chip using tmp fp2 chip
    fn create_dense_fp2_muldiv_chip(
        tester: &GpuChipTestBuilder,
        modulus: &num_bigint::BigUint,
        offset: usize,
        bitwise_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> NewVmChipWrapper<F, Fp2Air<2, NUM_LIMBS>, Fp2Step<2, NUM_LIMBS>, DenseRecordArena> {
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        // build a temporary sparse chip to pull air & step
        let tmp_chip = openvm_algebra_circuit::fp2_chip::Fp2MulDivChip::<F, 2, NUM_LIMBS>::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            tester.cpu_memory_helper(),
            tester.address_bits(),
            config,
            offset,
            bitwise_chip.clone(),
            tester.cpu_range_checker(),
        );
        let mut dense_chip = NewVmChipWrapper::<F, _, _, DenseRecordArena>::new(
            tmp_chip.0.air,
            tmp_chip.0.step,
            tester.cpu_memory_helper(),
        );
        dense_chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        dense_chip
    }

    fn create_sparse_fp2_muldiv_chip(
        tester: &GpuChipTestBuilder,
        modulus: &num_bigint::BigUint,
        offset: usize,
        bitwise_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> NewVmChipWrapper<F, Fp2Air<2, NUM_LIMBS>, Fp2Step<2, NUM_LIMBS>, MatrixRecordArena<F>>
    {
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let chip = openvm_algebra_circuit::fp2_chip::Fp2MulDivChip::<F, 2, NUM_LIMBS>::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            tester.cpu_memory_helper(),
            tester.address_bits(),
            config,
            offset,
            bitwise_chip.clone(),
            tester.cpu_range_checker(),
        );
        let mut wrapped_chip = chip.0;
        wrapped_chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        wrapped_chip
    }

    #[test]
    fn test_fp2_muldiv_tracegen_mod_builder() {
        let modulus = secp256k1_coord_prime();
        let offset = Fp2Opcode::CLASS_OFFSET;
        let mut tester = GpuChipTestBuilder::default()
            .with_variable_range_checker()
            .with_bitwise_op_lookup(BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS));
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let shared_bitwise = SharedBitwiseOperationLookupChip::new(bitwise_bus);

        // Build dense CPU chip to generate records
        let mut dense_chip =
            create_dense_fp2_muldiv_chip(&tester, &modulus, offset, shared_bitwise.clone());
        // Build sparse CPU chip for expected trace
        let mut sparse_chip =
            create_sparse_fp2_muldiv_chip(&tester, &modulus, offset, shared_bitwise.clone());
        // GPU chip wrapper (arena set later)
        let mut gpu_chip = Fp2MulDivChipGpu::new(
            dense_chip.air.clone(),
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            None,
        );

        let mut rng = create_seeded_rng();
        for i in 0..50 {
            let is_setup = i == 0;

            let (a_c0, a_c1, b_c0, b_c1, op_local) = if is_setup {
                (
                    modulus.clone(),
                    num_bigint::BigUint::from(0u32),
                    num_bigint::BigUint::from(0u32),
                    num_bigint::BigUint::from(0u32),
                    Fp2Opcode::SETUP_MULDIV as usize,
                )
            } else {
                // Generate random Fp2 elements (c0, c1)
                let a_c0_digits: Vec<_> = (0..NUM_LIMBS)
                    .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
                    .collect();
                let mut a_c0 = num_bigint::BigUint::new(a_c0_digits);
                let a_c1_digits: Vec<_> = (0..NUM_LIMBS)
                    .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
                    .collect();
                let mut a_c1 = num_bigint::BigUint::new(a_c1_digits);

                let b_c0_digits: Vec<_> = (0..NUM_LIMBS)
                    .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
                    .collect();
                let mut b_c0 = num_bigint::BigUint::new(b_c0_digits);
                let b_c1_digits: Vec<_> = (0..NUM_LIMBS)
                    .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
                    .collect();
                let mut b_c1 = num_bigint::BigUint::new(b_c1_digits);

                let op = rng.gen_range(0..2);
                a_c0 %= &modulus;
                a_c1 %= &modulus;
                b_c0 %= &modulus;
                b_c1 %= &modulus;

                // For DIV operation, make sure b is not zero (both components can't be zero)
                if op == 1
                    && b_c0 == num_bigint::BigUint::from(0u32)
                    && b_c1 == num_bigint::BigUint::from(0u32)
                {
                    b_c0 = num_bigint::BigUint::from(1u32);
                }

                let op_local = if op == 0 {
                    Fp2Opcode::MUL as usize
                } else {
                    Fp2Opcode::DIV as usize
                };
                (a_c0, a_c1, b_c0, b_c1, op_local)
            };

            let opcode = offset + op_local;

            // Convert to limbs - Fp2 elements are stored as [c0, c1] for each operand
            let a_c0_limbs: [BabyBear; NUM_LIMBS] =
                biguint_to_limbs(a_c0, LIMB_BITS).map(BabyBear::from_canonical_u32);
            let a_c1_limbs: [BabyBear; NUM_LIMBS] =
                biguint_to_limbs(a_c1, LIMB_BITS).map(BabyBear::from_canonical_u32);
            let b_c0_limbs: [BabyBear; NUM_LIMBS] =
                biguint_to_limbs(b_c0, LIMB_BITS).map(BabyBear::from_canonical_u32);
            let b_c1_limbs: [BabyBear; NUM_LIMBS] =
                biguint_to_limbs(b_c1, LIMB_BITS).map(BabyBear::from_canonical_u32);

            let ptr_as = 1;
            let data_as = 2;

            // Three pointers for registers: rs1 (operand a), rs2 (operand b), rd (result)
            let rs1_ptr = 0;
            let rs2_ptr = 3 * openvm_rv32im_circuit::adapters::RV32_REGISTER_NUM_LIMBS;
            let rd_ptr = 6 * openvm_rv32im_circuit::adapters::RV32_REGISTER_NUM_LIMBS;

            // Memory addresses for Fp2 elements (each Fp2 has 2 field elements)
            // Using increment of 128 to match CPU test's rv32_write_heap_default
            let a_base_addr = 0u32;
            let b_base_addr = 128u32;
            let result_base_addr = 256u32;

            // Write pointers to registers
            tester.write(
                ptr_as,
                rs1_ptr,
                a_base_addr.to_le_bytes().map(BabyBear::from_canonical_u8),
            );
            tester.write(
                ptr_as,
                rs2_ptr,
                b_base_addr.to_le_bytes().map(BabyBear::from_canonical_u8),
            );
            tester.write(
                ptr_as,
                rd_ptr,
                result_base_addr
                    .to_le_bytes()
                    .map(BabyBear::from_canonical_u8),
            );

            // Write operand data - Fp2 elements stored as consecutive field elements
            tester.write(data_as, a_base_addr as usize, a_c0_limbs);
            tester.write(
                data_as,
                (a_base_addr + NUM_LIMBS as u32) as usize,
                a_c1_limbs,
            );
            tester.write(data_as, b_base_addr as usize, b_c0_limbs);
            tester.write(
                data_as,
                (b_base_addr + NUM_LIMBS as u32) as usize,
                b_c1_limbs,
            );

            let instruction = Instruction::from_isize(
                VmOpcode::from_usize(opcode),
                rd_ptr as isize,
                rs1_ptr as isize,
                rs2_ptr as isize,
                ptr_as as isize,
                data_as as isize,
            );
            tester.execute(&mut dense_chip, &instruction);
        }

        // Transfer records from dense arena to sparse chip matrix arena
        type Record<'a> = (
            &'a mut Rv32VecHeapAdapterRecord<2, 2, 2, NUM_LIMBS, NUM_LIMBS>,
            openvm_mod_circuit_builder::FieldExpressionCoreRecordMut<'a>,
        );
        dense_chip
            .arena
            .get_record_seeker::<Record<'_>, _>()
            .transfer_to_matrix_arena(
                &mut sparse_chip.arena,
                dense_chip.step.0.get_record_layout::<F>(),
            );

        // Assign arena to gpu chip
        gpu_chip.arena = Some(&dense_chip.arena);

        tester
            .build()
            .load_and_compare(gpu_chip, sparse_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
