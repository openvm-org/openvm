use std::{cell::RefCell, rc::Rc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::var_range::VariableRangeCheckerBus;
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, ExprBuilderConfig, FieldExpr, SymbolicExpr},
    field_extension::Fp2,
};
use axvm_circuit_derive::InstructionExecutor;
use p3_field::PrimeField32;

use crate::{
    arch::{instructions::Fp2Opcode, VmChipWrapper},
    intrinsics::field_expression::FieldExpressionCoreChip,
    rv32im::adapters::Rv32VecHeapAdapterChip,
    system::memory::MemoryControllerRef,
};

// Input: Fp2 * 2
// Output: Fp2
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct Fp2MulDivChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    Fp2MulDivChip<F, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        memory_controller: MemoryControllerRef<F>,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> Self {
        let (expr, flag_id) =
            fp2_muldiv_expr(config, memory_controller.borrow().range_checker.bus());
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![Fp2Opcode::MUL as usize, Fp2Opcode::DIV as usize],
            vec![flag_id],
            memory_controller.borrow().range_checker.clone(),
            "Fp2MulDiv",
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}

pub fn fp2_muldiv_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
) -> (FieldExpr, usize) {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let x = Fp2::new(builder.clone());
    let mut y = Fp2::new(builder.clone());
    let flag = builder.borrow_mut().new_flag();
    let (z_idx, mut z) = Fp2::new_var(builder.clone());
    let mut lvar = Fp2::select(flag, &x, &z);
    let mut rvar = Fp2::select(flag, &z, &x);
    let fp2_constraint = lvar.mul(&mut y).sub(&mut rvar);

    z.save_output();
    builder
        .borrow_mut()
        .set_constraint(z_idx.0, fp2_constraint.c0.expr);
    builder
        .borrow_mut()
        .set_constraint(z_idx.1, fp2_constraint.c1.expr);

    // Compute expression has to be done manually at the SymbolicExpr level.
    // Otherwise it saves the quotient and introduces new variables.
    let compute_z0_div = (&x.c0.expr * &y.c0.expr + &x.c1.expr * &y.c1.expr)
        / (&y.c0.expr * &y.c0.expr + &y.c1.expr * &y.c1.expr);
    let compute_z0_mul = &x.c0.expr * &y.c0.expr - &x.c1.expr * &y.c1.expr;
    let compute_z0 = SymbolicExpr::Select(flag, Box::new(compute_z0_mul), Box::new(compute_z0_div));
    let compute_z1_div = (&x.c1.expr * &y.c0.expr - &x.c0.expr * &y.c1.expr)
        / (&y.c0.expr * &y.c0.expr + &y.c1.expr * &y.c1.expr);
    let compute_z1_mul = &x.c1.expr * &y.c0.expr + &x.c0.expr * &y.c1.expr;
    let compute_z1 = SymbolicExpr::Select(flag, Box::new(compute_z1_mul), Box::new(compute_z1_div));
    builder.borrow_mut().set_compute(z_idx.0, compute_z0);
    builder.borrow_mut().set_compute(z_idx.1, compute_z1);

    let builder = builder.borrow().clone();
    (FieldExpr::new(builder, range_bus), flag)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ax_circuit_primitives::bitwise_op_lookup::{
        BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    };
    use ax_ecc_primitives::{
        field_expression::ExprBuilderConfig,
        test_utils::{bn254_fq2_to_biguint_vec, bn254_fq_to_biguint},
    };
    use axvm_ecc_constants::BN254;
    use axvm_instructions::{riscv::RV32_CELL_BITS, UsizeOpcode};
    use halo2curves_axiom::{bn256::Fq2, ff::Field};
    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use rand::{rngs::StdRng, SeedableRng};

    use super::fp2_muldiv_expr;
    use crate::{
        arch::{
            instructions::Fp2Opcode, testing::VmChipTestBuilder, VmChipWrapper,
            BITWISE_OP_LOOKUP_BUS,
        },
        intrinsics::field_expression::FieldExpressionCoreChip,
        rv32im::adapters::Rv32VecHeapAdapterChip,
        utils::{biguint_to_limbs, rv32_write_heap_default},
    };

    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    type F = BabyBear;

    #[test]
    fn test_fp2_muldiv() {
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: BN254.MODULUS.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let (expr, flag_idx) = fp2_muldiv_expr(
            config,
            tester.memory_controller().borrow().range_checker.bus(),
        );

        let core = FieldExpressionCoreChip::new(
            expr,
            Fp2Opcode::default_offset(),
            vec![Fp2Opcode::MUL as usize, Fp2Opcode::DIV as usize],
            vec![flag_idx],
            tester.memory_controller().borrow().range_checker.clone(),
            "Fp2MulDiv",
        );
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
        let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

        let mut rng = StdRng::seed_from_u64(42);
        let x = Fq2::random(&mut rng);
        let y = Fq2::random(&mut rng);
        let inputs = [x.c0, x.c1, y.c0, y.c1].map(|x| bn254_fq_to_biguint(&x));

        let expected_mul = bn254_fq2_to_biguint_vec(&(x * y));
        let r_mul = chip
            .core
            .expr()
            .execute_with_output(inputs.to_vec(), vec![true]);
        assert_eq!(r_mul.len(), 2);
        assert_eq!(r_mul[0], expected_mul[0]);
        assert_eq!(r_mul[1], expected_mul[1]);

        let expected_div = bn254_fq2_to_biguint_vec(&(x * y.invert().unwrap()));
        let r_div = chip
            .core
            .expr()
            .execute_with_output(inputs.to_vec(), vec![false]);
        assert_eq!(r_div.len(), 2);
        assert_eq!(r_div[0], expected_div[0]);
        assert_eq!(r_div[1], expected_div[1]);

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
            chip.core.air.offset + Fp2Opcode::MUL as usize,
        );
        let instruction2 = rv32_write_heap_default(
            &mut tester,
            x_limbs,
            y_limbs,
            chip.core.air.offset + Fp2Opcode::DIV as usize,
        );
        tester.execute(&mut chip, instruction1);
        tester.execute(&mut chip, instruction2);
        let tester = tester.build().load(chip).load(bitwise_chip).finalize();
        tester.simple_test().expect("Verification failed");
    }
}
