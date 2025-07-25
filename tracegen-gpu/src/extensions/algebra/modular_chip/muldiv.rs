use std::sync::Arc;

use derive_new::new;
use openvm_algebra_circuit::modular_chip::muldiv_expr;
use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
use openvm_circuit::arch::{AdapterCoreLayout, DenseRecordArena, RecordSeeker};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::{
    ExprBuilderConfig, FieldExpressionCoreAir, FieldExpressionMetadata,
};
use openvm_rv32_adapters::{Rv32VecHeapAdapterCols, Rv32VecHeapAdapterStep};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend};

use crate::{
    extensions::algebra::AlgebraRecord,
    get_empty_air_proving_ctx,
    mod_builder::field_expression::FieldExpressionChipGPU,
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
};

#[derive(new)]
pub struct ModularMulDivChipGpu<const BLOCKS: usize, const BLOCK_SIZE: usize> {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub config: ExprBuilderConfig,
    pub offset: usize,
    pub pointer_max_bits: u32,
    pub timestamp_max_bits: u32,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for ModularMulDivChipGpu<BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let range_bus = self.range_checker.cpu_chip.as_ref().unwrap().bus();
        let (expr, is_mul_flag, is_div_flag) = muldiv_expr(self.config.clone(), range_bus);

        let total_input_limbs = expr.builder.num_input * expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterStep<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size = RecordSeeker::<
            DenseRecordArena,
            AlgebraRecord<2, BLOCKS, BLOCK_SIZE>,
            _,
        >::get_aligned_record_size(&layout);

        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;

        let local_opcode_idx = vec![
            Rv32ModularArithmeticOpcode::MUL as usize,
            Rv32ModularArithmeticOpcode::DIV as usize,
            Rv32ModularArithmeticOpcode::SETUP_MULDIV as usize,
        ];
        let opcode_flag_idx = vec![is_mul_flag, is_div_flag];

        let air = FieldExpressionCoreAir::new(expr, self.offset, local_opcode_idx, opcode_flag_idx);

        let adapter_width =
            Rv32VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width();

        let d_records = records.to_device().unwrap();

        let field_expr_chip = FieldExpressionChipGPU::new(
            air,
            d_records,
            num_records,
            record_size,
            adapter_width,
            BLOCKS,
            self.range_checker.clone(),
            self.bitwise_lookup.clone(),
            self.pointer_max_bits,
            self.timestamp_max_bits,
        );

        let d_trace = field_expr_chip.generate_field_trace();

        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use num_traits::Zero;
    use openvm_algebra_circuit::modular_chip::{
        get_modular_muldiv_air, get_modular_muldiv_chip, get_modular_muldiv_step, ModularAir,
        ModularChip, ModularStep,
    };
    use openvm_circuit::arch::testing::memory::gen_pointer;
    use openvm_circuit_primitives::{
        bigint::utils::secp256k1_coord_prime, bitwise_op_lookup::BitwiseOperationLookupChip,
        var_range::VariableRangeCheckerChip,
    };
    use openvm_instructions::{instruction::Instruction, LocalOpcode, VmOpcode};
    use openvm_mod_circuit_builder::test_utils::biguint_to_limbs;
    use openvm_rv32im_circuit::adapters::RV32_REGISTER_NUM_LIMBS;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};

    use super::*;
    use crate::testing::{
        default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
        GpuTestChipHarness,
    };

    const LIMB_BITS: usize = 8;
    const MAX_INS_CAPACITY: usize = 128;

    fn create_test_harness<const BLOCKS: usize, const BLOCK_SIZE: usize>(
        tester: &GpuChipTestBuilder,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> GpuTestChipHarness<
        F,
        ModularStep<BLOCKS, BLOCK_SIZE>,
        ModularAir<BLOCKS, BLOCK_SIZE>,
        ModularMulDivChipGpu<BLOCKS, BLOCK_SIZE>,
        ModularChip<F, BLOCKS, BLOCK_SIZE>,
    > {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        let bitwise_bus = default_bitwise_lookup_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));

        let air = get_modular_muldiv_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            range_bus,
            bitwise_bus,
            tester.address_bits(),
            offset,
        );
        let executor =
            get_modular_muldiv_step(config.clone(), range_bus, tester.address_bits(), offset);

        let cpu_chip = get_modular_muldiv_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            dummy_bitwise_chip,
            tester.address_bits(),
        );
        let gpu_chip = ModularMulDivChipGpu::new(
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            config,
            offset,
            tester.address_bits() as u32,
            tester.timestamp_max_bits() as u32,
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute_muldiv<
        const BLOCKS: usize,
        const BLOCK_SIZE: usize,
        const NUM_LIMBS: usize,
    >(
        tester: &mut GpuChipTestBuilder,
        harness: &mut GpuTestChipHarness<
            F,
            ModularStep<BLOCKS, BLOCK_SIZE>,
            ModularAir<BLOCKS, BLOCK_SIZE>,
            ModularMulDivChipGpu<BLOCKS, BLOCK_SIZE>,
            ModularChip<F, BLOCKS, BLOCK_SIZE>,
        >,
        rng: &mut StdRng,
        modulus: &BigUint,
        is_setup: bool,
        offset: usize,
    ) {
        let (a, b, op) = if is_setup {
            (
                modulus.clone(),
                BigUint::zero(),
                Rv32ModularArithmeticOpcode::SETUP_MULDIV as usize,
            )
        } else {
            let a_digits: Vec<_> = (0..NUM_LIMBS)
                .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
                .collect();
            let mut a = BigUint::new(a_digits.clone());
            let b_digits: Vec<_> = (0..NUM_LIMBS)
                .map(|_| rng.gen_range(1..(1 << LIMB_BITS)))
                .collect();
            let mut b = BigUint::new(b_digits.clone());

            let op = rng.gen_range(0..2);
            let op = match op {
                0 => Rv32ModularArithmeticOpcode::MUL as usize,
                1 => Rv32ModularArithmeticOpcode::DIV as usize,
                _ => panic!(),
            };
            a %= modulus;
            b %= modulus;
            (a, b, op)
        };

        let ptr_as = 1;
        let addr_ptr1 = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let addr_ptr2 = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let addr_ptr3 = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

        let data_as = 2;
        let address1 = 0u32;
        let address2 = 128u32;
        let address3: u32 = (1 << 28) + 1228;

        tester.write::<RV32_REGISTER_NUM_LIMBS>(
            ptr_as,
            addr_ptr1,
            address1.to_le_bytes().map(F::from_canonical_u8),
        );
        tester.write::<RV32_REGISTER_NUM_LIMBS>(
            ptr_as,
            addr_ptr2,
            address2.to_le_bytes().map(F::from_canonical_u8),
        );
        tester.write::<RV32_REGISTER_NUM_LIMBS>(
            ptr_as,
            addr_ptr3,
            address3.to_le_bytes().map(F::from_canonical_u8),
        );

        let a_limbs = biguint_to_limbs::<NUM_LIMBS>(a, LIMB_BITS).map(F::from_canonical_u32);
        tester.write(data_as, address1 as usize, a_limbs);
        let b_limbs = biguint_to_limbs::<NUM_LIMBS>(b, LIMB_BITS).map(F::from_canonical_u32);
        tester.write(data_as, address2 as usize, b_limbs);

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(offset + op),
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );
        tester.execute(
            &mut harness.executor,
            &mut harness.dense_arena,
            &instruction,
        );
    }

    fn run_test_with_config<
        const BLOCKS: usize,
        const BLOCK_SIZE: usize,
        const NUM_LIMBS: usize,
    >(
        modulus: BigUint,
        num_ops: usize,
    ) {
        let mut rng = create_seeded_rng();

        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

        let offset = Rv32ModularArithmeticOpcode::CLASS_OFFSET;
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness = create_test_harness::<BLOCKS, BLOCK_SIZE>(&tester, config, offset);

        for i in 0..num_ops {
            set_and_execute_muldiv::<BLOCKS, BLOCK_SIZE, NUM_LIMBS>(
                &mut tester,
                &mut harness,
                &mut rng,
                &modulus,
                i == 0,
                offset,
            );
        }

        harness
            .dense_arena
            .get_record_seeker::<AlgebraRecord<2, BLOCKS, BLOCK_SIZE>, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                harness.executor.get_record_layout::<F>(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[test]
    fn test_modular_muldiv_gpu() {
        run_test_with_config::<1, 32, 32>(secp256k1_coord_prime(), 50);
    }
}
