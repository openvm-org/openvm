use std::{cell::RefCell, rc::Rc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::var_range::VariableRangeCheckerBus;
use ax_mod_circuit_builder::{ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreChip};
use axvm_algebra_transpiler::Fp2Opcode;
use axvm_circuit::{arch::VmChipWrapper, system::memory::MemoryControllerRef};
use axvm_circuit_derive::InstructionExecutor;
use axvm_rv32_adapters::Rv32VecHeapAdapterChip;
use p3_field::PrimeField32;

use crate::Fp2;

// Input: Fp2 * 2
// Output: Fp2
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct Fp2AddSubChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub  VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    Fp2AddSubChip<F, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        memory_controller: MemoryControllerRef<F>,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> Self {
        let (expr, is_add_flag, is_sub_flag) =
            fp2_addsub_expr(config, memory_controller.borrow().range_checker.bus());
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![
                Fp2Opcode::ADD as usize,
                Fp2Opcode::SUB as usize,
                Fp2Opcode::SETUP_ADDSUB as usize,
            ],
            vec![is_add_flag, is_sub_flag],
            memory_controller.borrow().range_checker.clone(),
            "Fp2AddSub",
            false,
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}

pub fn fp2_addsub_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
) -> (FieldExpr, usize, usize) {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x = Fp2::new(builder.clone());
    let mut y = Fp2::new(builder.clone());
    let add = x.add(&mut y);
    let sub = x.sub(&mut y);

    let is_add_flag = builder.borrow_mut().new_flag();
    let is_sub_flag = builder.borrow_mut().new_flag();
    let diff = Fp2::select(is_sub_flag, &sub, &x);
    let mut z = Fp2::select(is_add_flag, &add, &diff);
    z.save_output();

    let builder = builder.borrow().clone();
    (
        FieldExpr::new(builder, range_bus, true),
        is_add_flag,
        is_sub_flag,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ax_circuit_primitives::bitwise_op_lookup::{
        BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    };
    use ax_mod_circuit_builder::{
        test_utils::{biguint_to_limbs, bn254_fq2_to_biguint_vec, bn254_fq_to_biguint},
        ExprBuilderConfig,
    };
    use axvm_algebra_transpiler::Fp2Opcode;
    use axvm_circuit::arch::{testing::VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS};
    use axvm_instructions::{riscv::RV32_CELL_BITS, UsizeOpcode};
    use axvm_pairing_guest::bn254::BN254_MODULUS;
    use axvm_rv32_adapters::{rv32_write_heap_default, Rv32VecHeapAdapterChip};
    use halo2curves_axiom::{bn256::Fq2, ff::Field};
    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use rand::{rngs::StdRng, SeedableRng};

    use super::Fp2AddSubChip;

    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    type F = BabyBear;

    #[test]
    fn test_fp2_addsub() {
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: BN254_MODULUS.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));
        let adapter = Rv32VecHeapAdapterChip::<F, 2, 2, 2, NUM_LIMBS, NUM_LIMBS>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
            bitwise_chip.clone(),
        );
        let mut chip = Fp2AddSubChip::new(
            adapter,
            tester.memory_controller(),
            config,
            Fp2Opcode::default_offset(),
        );

        let mut rng = StdRng::seed_from_u64(42);
        let x = Fq2::random(&mut rng);
        let y = Fq2::random(&mut rng);
        let inputs = [x.c0, x.c1, y.c0, y.c1].map(bn254_fq_to_biguint);

        let expected_sum = bn254_fq2_to_biguint_vec(x + y);
        let r_sum = chip
            .0
            .core
            .expr()
            .execute_with_output(inputs.to_vec(), vec![true, false]);
        assert_eq!(r_sum.len(), 2);
        assert_eq!(r_sum[0], expected_sum[0]);
        assert_eq!(r_sum[1], expected_sum[1]);

        let expected_sub = bn254_fq2_to_biguint_vec(x - y);
        let r_sub = chip
            .0
            .core
            .expr()
            .execute_with_output(inputs.to_vec(), vec![false, true]);
        assert_eq!(r_sub.len(), 2);
        assert_eq!(r_sub[0], expected_sub[0]);
        assert_eq!(r_sub[1], expected_sub[1]);

        let x_limbs = inputs[0..2]
            .iter()
            .map(|x| {
                biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS)
                    .map(BabyBear::from_canonical_u32)
            })
            .collect_vec();
        let y_limbs = inputs[2..4]
            .iter()
            .map(|x| {
                biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS)
                    .map(BabyBear::from_canonical_u32)
            })
            .collect_vec();
        let instruction1 = rv32_write_heap_default(
            &mut tester,
            x_limbs.clone(),
            y_limbs.clone(),
            chip.0.core.air.offset + Fp2Opcode::ADD as usize,
        );
        let instruction2 = rv32_write_heap_default(
            &mut tester,
            x_limbs,
            y_limbs,
            chip.0.core.air.offset + Fp2Opcode::SUB as usize,
        );
        tester.execute(&mut chip, instruction1);
        tester.execute(&mut chip, instruction2);
        let tester = tester.build().load(chip).load(bitwise_chip).finalize();
        tester.simple_test().expect("Verification failed");
    }
}
