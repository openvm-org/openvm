use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use openvm_algebra_transpiler::Fp2Opcode;
use openvm_circuit::{arch::VmChipWrapper, system::memory::OfflineMemory};
use openvm_circuit_derive::InstructionExecutor;
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerBus,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreChip, SymbolicExpr,
};
use openvm_rv32_adapters::Rv32VecHeapAdapterChip;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::Fp2;

// Input: Fp2 * 2
// Output: Fp2
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct Fp2MulDivChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub  VmChipWrapper<
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
        config: ExprBuilderConfig,
        offset: usize,
        range_checker: SharedVariableRangeCheckerChip,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    ) -> Self {
        let (expr, is_mul_flag, is_div_flag) = fp2_muldiv_expr(config, range_checker.bus());
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![
                Fp2Opcode::MUL as usize,
                Fp2Opcode::DIV as usize,
                Fp2Opcode::SETUP_MULDIV as usize,
            ],
            vec![is_mul_flag, is_div_flag],
            range_checker,
            "Fp2MulDiv",
            false,
        );
        Self(VmChipWrapper::new(adapter, core, offline_memory))
    }
}

pub fn fp2_muldiv_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
) -> (FieldExpr, usize, usize) {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let x = Fp2::new(builder.clone());
    let mut y = Fp2::new(builder.clone());
    let is_mul_flag = builder.borrow_mut().new_flag();
    let is_div_flag = builder.borrow_mut().new_flag();
    let (z_idx, mut z) = Fp2::new_var(builder.clone());

    let mut lvar = Fp2::select(is_mul_flag, &x, &z);

    let mut rvar = Fp2::select(is_mul_flag, &z, &x);
    let fp2_constraint = lvar.mul(&mut y).sub(&mut rvar);
    // When it's SETUP op, the constraints is z * y - x = 0, it still works as:
    // x.c0 = x.c1 = p == 0, y.c0 = y.c1 = 0, so whatever z is, z * 0 - 0 = 0

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
    let compute_z0 = SymbolicExpr::Select(
        is_mul_flag,
        Box::new(compute_z0_mul),
        Box::new(SymbolicExpr::Select(
            is_div_flag,
            Box::new(compute_z0_div),
            Box::new(x.c0.expr.clone()),
        )),
    );
    let compute_z1_div = (&x.c1.expr * &y.c0.expr - &x.c0.expr * &y.c1.expr)
        / (&y.c0.expr * &y.c0.expr + &y.c1.expr * &y.c1.expr);
    let compute_z1_mul = &x.c1.expr * &y.c0.expr + &x.c0.expr * &y.c1.expr;
    let compute_z1 = SymbolicExpr::Select(
        is_mul_flag,
        Box::new(compute_z1_mul),
        Box::new(SymbolicExpr::Select(
            is_div_flag,
            Box::new(compute_z1_div),
            Box::new(x.c1.expr),
        )),
    );
    builder.borrow_mut().set_compute(z_idx.0, compute_z0);
    builder.borrow_mut().set_compute(z_idx.1, compute_z1);

    let builder = builder.borrow().clone();
    (
        FieldExpr::new(builder, range_bus, true),
        is_mul_flag,
        is_div_flag,
    )
}

#[cfg(test)]
mod tests {

    use halo2curves_axiom::{bn256::Fq2, ff::Field};
    use itertools::Itertools;
    use openvm_algebra_transpiler::Fp2Opcode;
    use openvm_circuit::arch::testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS};
    use openvm_circuit_primitives::bitwise_op_lookup::{
        BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
    };
    use openvm_instructions::{riscv::RV32_CELL_BITS, LocalOpcode};
    use openvm_mod_circuit_builder::{
        test_utils::{biguint_to_limbs, bn254_fq2_to_biguint_vec, bn254_fq_to_biguint},
        ExprBuilderConfig,
    };
    use openvm_pairing_guest::bn254::BN254_MODULUS;
    use openvm_rv32_adapters::{rv32_write_heap_default, Rv32VecHeapAdapterChip};
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use rand::{rngs::StdRng, SeedableRng};

    use super::Fp2MulDivChip;

    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    type F = BabyBear;

    #[test]
    fn test_fp2_muldiv() {
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let modulus = BN254_MODULUS.clone();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
        let adapter = Rv32VecHeapAdapterChip::<F, 2, 2, 2, NUM_LIMBS, NUM_LIMBS>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_bridge(),
            tester.address_bits(),
            bitwise_chip.clone(),
        );
        let mut chip = Fp2MulDivChip::new(
            adapter,
            config,
            Fp2Opcode::CLASS_OFFSET,
            tester.range_checker(),
            tester.offline_memory_mutex_arc(),
        );
        assert_eq!(
            chip.0.core.expr().builder.num_variables,
            2,
            "Fp2MulDiv should only introduce new z Fp2 variable (2 Fp var)"
        );

        let mut rng = StdRng::seed_from_u64(42);
        let x = Fq2::random(&mut rng);
        let y = Fq2::random(&mut rng);
        let inputs = [x.c0, x.c1, y.c0, y.c1].map(bn254_fq_to_biguint);

        let expected_mul = bn254_fq2_to_biguint_vec(x * y);
        let r_mul = chip
            .0
            .core
            .expr()
            .execute_with_output(inputs.to_vec(), vec![true, false]);
        assert_eq!(r_mul.len(), 2);
        assert_eq!(r_mul[0], expected_mul[0]);
        assert_eq!(r_mul[1], expected_mul[1]);

        let expected_div = bn254_fq2_to_biguint_vec(x * y.invert().unwrap());
        let r_div = chip
            .0
            .core
            .expr()
            .execute_with_output(inputs.to_vec(), vec![false, true]);
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
        let modulus =
            biguint_to_limbs::<NUM_LIMBS>(modulus, LIMB_BITS).map(BabyBear::from_canonical_u32);
        let zero = [BabyBear::ZERO; NUM_LIMBS];
        let setup_instruction = rv32_write_heap_default(
            &mut tester,
            vec![modulus, zero],
            vec![zero; 2],
            chip.0.core.air.offset + Fp2Opcode::SETUP_MULDIV as usize,
        );
        let instruction1 = rv32_write_heap_default(
            &mut tester,
            x_limbs.clone(),
            y_limbs.clone(),
            chip.0.core.air.offset + Fp2Opcode::MUL as usize,
        );
        let instruction2 = rv32_write_heap_default(
            &mut tester,
            x_limbs,
            y_limbs,
            chip.0.core.air.offset + Fp2Opcode::DIV as usize,
        );
        tester.execute(&mut chip, &setup_instruction);
        tester.execute(&mut chip, &instruction1);
        tester.execute(&mut chip, &instruction2);
        let tester = tester.build().load(chip).load(bitwise_chip).finalize();
        tester.simple_test().expect("Verification failed");
    }
}
