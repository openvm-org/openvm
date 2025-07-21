use std::sync::Arc;

use derive_new::new;
use openvm_circuit::arch::{AdapterCoreLayout, DenseRecordArena, RecordSeeker};
use openvm_mod_circuit_builder::FieldExpressionMetadata;
use openvm_rv32_adapters::{Rv32VecHeapAdapterRecord, Rv32VecHeapAdapterStep};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend, types::SC,
};

use super::WeierstrassAir;
use crate::{
    mod_builder::field_expression::{constants::LIMB_BITS, FieldExpressionChipGPU},
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
    DeviceChip,
};

#[derive(new)]
pub struct EcDoubleChipGpu<'a, const BLOCKS: usize, const BLOCK_SIZE: usize> {
    pub air: WeierstrassAir<1, BLOCKS, BLOCK_SIZE>,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<LIMB_BITS>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> EcDoubleChipGpu<'_, BLOCKS, BLOCK_SIZE> {
    fn get_record_size(&self) -> usize {
        let total_input_limbs =
            self.air.core.expr.builder.num_input * self.air.core.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterStep<1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        RecordSeeker::<
            DenseRecordArena,
            (
                &mut Rv32VecHeapAdapterRecord<1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
                openvm_mod_circuit_builder::FieldExpressionCoreRecordMut<'_>,
            ),
            _,
        >::get_aligned_record_size(&layout)
    }
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> ChipUsageGetter
    for EcDoubleChipGpu<'_, BLOCKS, BLOCK_SIZE>
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
    for EcDoubleChipGpu<'_, BLOCKS, BLOCK_SIZE>
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let buf = &self.arena.unwrap().allocated();
        let d_records = buf.to_device().unwrap();

        let record_size = self.get_record_size();
        let num_records = buf.len() / record_size;

        let adapter_width = <openvm_rv32_adapters::Rv32VecHeapAdapterAir<
            1,
            BLOCKS,
            BLOCKS,
            BLOCK_SIZE,
            BLOCK_SIZE,
        > as BaseAir<F>>::width(&self.air.adapter);
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
    use openvm_circuit::arch::{
        testing::BITWISE_OP_LOOKUP_BUS, DenseRecordArena, MatrixRecordArena, NewVmChipWrapper,
    };
    use openvm_circuit_primitives::{
        bigint::utils::secp256k1_coord_prime,
        bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    };
    use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
    use openvm_instructions::{
        instruction::Instruction, riscv::RV32_CELL_BITS, LocalOpcode, VmOpcode,
    };
    use openvm_mod_circuit_builder::{test_utils::biguint_to_limbs, ExprBuilderConfig};
    use openvm_rv32_adapters::Rv32VecHeapAdapterRecord;
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use stark_backend_gpu::prelude::F;

    use super::{
        super::{WeierstrassAir, WeierstrassStep},
        *,
    };
    use crate::testing::GpuChipTestBuilder;

    // Use BLOCKS = 2, BLOCK_SIZE = NUM_LIMBS for secp256k1 points
    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    const MAX_INS_CAPACITY: usize = 512;

    // Helper to build dense chip using tmp ec double chip
    fn create_dense_ec_double_chip(
        tester: &GpuChipTestBuilder,
        modulus: &num_bigint::BigUint,
        offset: usize,
        bitwise_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        a_biguint: num_bigint::BigUint,
    ) -> NewVmChipWrapper<
        F,
        WeierstrassAir<1, 2, NUM_LIMBS>,
        WeierstrassStep<1, 2, NUM_LIMBS>,
        DenseRecordArena,
    > {
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        // build a temporary sparse chip to pull air & step
        let tmp_chip = openvm_ecc_circuit::EcDoubleChip::<F, 2, NUM_LIMBS>::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            tester.cpu_memory_helper(),
            tester.address_bits(),
            config,
            offset,
            bitwise_chip.clone(),
            tester.cpu_range_checker(),
            a_biguint,
        );
        let mut dense_chip = NewVmChipWrapper::<F, _, _, DenseRecordArena>::new(
            tmp_chip.0.air,
            tmp_chip.0.step,
            tester.cpu_memory_helper(),
        );
        dense_chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        dense_chip
    }

    fn create_sparse_ec_double_chip(
        tester: &GpuChipTestBuilder,
        modulus: &num_bigint::BigUint,
        offset: usize,
        bitwise_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        a_biguint: num_bigint::BigUint,
    ) -> NewVmChipWrapper<
        F,
        WeierstrassAir<1, 2, NUM_LIMBS>,
        WeierstrassStep<1, 2, NUM_LIMBS>,
        MatrixRecordArena<F>,
    > {
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let chip = openvm_ecc_circuit::EcDoubleChip::<F, 2, NUM_LIMBS>::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            tester.cpu_memory_helper(),
            tester.address_bits(),
            config,
            offset,
            bitwise_chip.clone(),
            tester.cpu_range_checker(),
            a_biguint,
        );
        let mut wrapped_chip = chip.0;
        wrapped_chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        wrapped_chip
    }

    #[test]
    fn test_ec_double_tracegen_mod_builder() {
        let modulus = secp256k1_coord_prime();
        let offset = Rv32WeierstrassOpcode::CLASS_OFFSET;
        // For secp256k1, a = 0
        let a_biguint = num_bigint::BigUint::from(0u32);

        let mut tester = GpuChipTestBuilder::default()
            .with_variable_range_checker()
            .with_bitwise_op_lookup(BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS));
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let shared_bitwise = SharedBitwiseOperationLookupChip::new(bitwise_bus);

        // Build dense CPU chip to generate records
        let mut dense_chip = create_dense_ec_double_chip(
            &tester,
            &modulus,
            offset,
            shared_bitwise.clone(),
            a_biguint.clone(),
        );
        // Build sparse CPU chip for expected trace
        let mut sparse_chip = create_sparse_ec_double_chip(
            &tester,
            &modulus,
            offset,
            shared_bitwise.clone(),
            a_biguint.clone(),
        );
        // GPU chip wrapper (arena set later)
        let mut gpu_chip = EcDoubleChipGpu::new(
            dense_chip.air.clone(),
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            None,
        );

        // Use known valid EC point from secp256k1 curve
        use num_traits::Num;
        let test_point = (
            num_bigint::BigUint::from(2u32),
            num_bigint::BigUint::from_str_radix(
                "69211104694897500952317515077652022726490027694212560352756646854116994689233",
                10,
            )
            .unwrap(),
        );

        for i in 0..5 {
            let is_setup = i == 0;

            let (x1, y1, op_local) = if is_setup {
                // Setup expects: prime and curve coefficient a
                (
                    modulus.clone(),
                    a_biguint.clone(),
                    Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
                )
            } else {
                // Use known valid point on secp256k1 curve
                let (p1_x, p1_y) = test_point.clone();

                let op_local = Rv32WeierstrassOpcode::EC_DOUBLE as usize;
                (p1_x, p1_y, op_local)
            };

            let opcode = offset + op_local;

            // Convert to limbs
            let x1_limbs: [BabyBear; NUM_LIMBS] =
                biguint_to_limbs(x1, LIMB_BITS).map(BabyBear::from_canonical_u32);
            let y1_limbs: [BabyBear; NUM_LIMBS] =
                biguint_to_limbs(y1, LIMB_BITS).map(BabyBear::from_canonical_u32);

            let ptr_as = 1;
            let data_as = 2;

            // Two pointers for registers: rs1 (input point), rd (result)
            let rs1_ptr = 0;
            let rd_ptr = 3 * openvm_rv32im_circuit::adapters::RV32_REGISTER_NUM_LIMBS;

            // Memory addresses for points (each point has 2 field elements)
            let p1_base_addr = 0u32;
            let result_base_addr = 128u32;

            // Write pointers to registers
            tester.write(
                ptr_as,
                rs1_ptr,
                p1_base_addr.to_le_bytes().map(BabyBear::from_canonical_u8),
            );
            tester.write(
                ptr_as,
                rd_ptr,
                result_base_addr
                    .to_le_bytes()
                    .map(BabyBear::from_canonical_u8),
            );

            // Write point data
            tester.write(data_as, p1_base_addr as usize, x1_limbs);
            tester.write(
                data_as,
                (p1_base_addr + NUM_LIMBS as u32) as usize,
                y1_limbs,
            );

            let instruction = Instruction::from_isize(
                VmOpcode::from_usize(opcode),
                rd_ptr as isize,
                rs1_ptr as isize,
                0, // rs2 not used for double
                ptr_as as isize,
                data_as as isize,
            );
            tester.execute(&mut dense_chip, &instruction);
        }

        // Transfer records from dense arena to sparse chip matrix arena
        type Record<'a> = (
            &'a mut Rv32VecHeapAdapterRecord<1, 2, 2, NUM_LIMBS, NUM_LIMBS>,
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
