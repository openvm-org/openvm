use std::{cell::RefCell, rc::Rc};

use openvm_algebra_transpiler::Fp2Opcode;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_derive::{InsExecutorE1, InsExecutorE2, InstructionExecutor};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    Chip, ChipUsageGetter,
};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir, SymbolicExpr,
};
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterStep};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{Fp2Air, Fp2Chip, Fp2Step};
use crate::Fp2;

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

// Input: Fp2 * 2
// Output: Fp2
#[derive(Chip, ChipUsageGetter, InstructionExecutor, InsExecutorE1, InsExecutorE2)]
pub struct Fp2MulDivChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub Fp2Chip<F, BLOCKS, BLOCK_SIZE>,
);

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    Fp2MulDivChip<F, BLOCKS, BLOCK_SIZE>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        execution_bridge: ExecutionBridge,
        memory_bridge: MemoryBridge,
        mem_helper: SharedMemoryHelper<F>,
        pointer_max_bits: usize,
        config: ExprBuilderConfig,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        range_checker: SharedVariableRangeCheckerChip,
    ) -> Self {
        let (expr, is_mul_flag, is_div_flag) = fp2_muldiv_expr(config, range_checker.bus());

        let local_opcode_idx = vec![
            Fp2Opcode::MUL as usize,
            Fp2Opcode::DIV as usize,
            Fp2Opcode::SETUP_MULDIV as usize,
        ];
        let opcode_flag_idx = vec![is_mul_flag, is_div_flag];
        let air = Fp2Air::new(
            Rv32VecHeapAdapterAir::new(
                execution_bridge,
                memory_bridge,
                bitwise_lookup_chip.bus(),
                pointer_max_bits,
            ),
            FieldExpressionCoreAir::new(
                expr.clone(),
                offset,
                local_opcode_idx.clone(),
                opcode_flag_idx.clone(),
            ),
        );

        let step = Fp2Step::new(
            Rv32VecHeapAdapterStep::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            offset,
            local_opcode_idx,
            opcode_flag_idx,
            range_checker,
            "Fp2MulDiv",
            false,
        );
        Self(Fp2Chip::new(air, step, mem_helper))
    }

    pub fn expr(&self) -> &FieldExpr {
        &self.0.step.0.expr
    }
}

#[cfg(test)]
mod tests {

    use halo2curves_axiom::{bn256::Fq2, ff::Field};
    use itertools::Itertools;
    use num_bigint::BigUint;
    use openvm_algebra_transpiler::Fp2Opcode;
    use openvm_circuit::arch::{
        testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        InsExecutorE1,
    };
    use openvm_circuit_primitives::bitwise_op_lookup::{
        BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
    };
    use openvm_instructions::{riscv::RV32_CELL_BITS, LocalOpcode};
    use openvm_mod_circuit_builder::{
        test_utils::{biguint_to_limbs, bn254_fq2_to_biguint_vec, bn254_fq_to_biguint},
        ExprBuilderConfig,
    };
    use openvm_pairing_guest::bn254::BN254_MODULUS;
    use openvm_rv32_adapters::rv32_write_heap_default;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};

    use crate::fp2_chip::Fp2MulDivChip;

    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    const OFFSET: usize = Fp2Opcode::CLASS_OFFSET;
    const MAX_INS_CAPACITY: usize = 128;
    type F = BabyBear;

    fn set_and_execute_rand(
        tester: &mut VmChipTestBuilder<F>,
        chip: &mut Fp2MulDivChip<F, 2, NUM_LIMBS>,
        modulus: &BigUint,
    ) {
        let mut rng = create_seeded_rng();
        let x = Fq2::random(&mut rng);
        let y = Fq2::random(&mut rng);
        let inputs = [x.c0, x.c1, y.c0, y.c1].map(bn254_fq_to_biguint);

        let expected_mul = bn254_fq2_to_biguint_vec(x * y);
        let r_mul = chip
            .expr()
            .execute_with_output(inputs.to_vec(), vec![true, false]);
        assert_eq!(r_mul.len(), 2);
        assert_eq!(r_mul[0], expected_mul[0]);
        assert_eq!(r_mul[1], expected_mul[1]);

        let expected_div = bn254_fq2_to_biguint_vec(x * y.invert().unwrap());
        let r_div = chip
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
        let modulus = biguint_to_limbs::<NUM_LIMBS>(modulus.clone(), LIMB_BITS)
            .map(BabyBear::from_canonical_u32);
        let zero = [BabyBear::ZERO; NUM_LIMBS];
        let setup_instruction = rv32_write_heap_default(
            tester,
            vec![modulus, zero],
            vec![zero; 2],
            OFFSET + Fp2Opcode::SETUP_MULDIV as usize,
        );
        let instruction1 = rv32_write_heap_default(
            tester,
            x_limbs.clone(),
            y_limbs.clone(),
            OFFSET + Fp2Opcode::MUL as usize,
        );
        let instruction2 =
            rv32_write_heap_default(tester, x_limbs, y_limbs, OFFSET + Fp2Opcode::DIV as usize);
        tester.execute(chip, &setup_instruction);
        tester.execute(chip, &instruction1);
        tester.execute(chip, &instruction2);
    }

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

        let mut chip = Fp2MulDivChip::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            tester.memory_helper(),
            tester.address_bits(),
            config,
            OFFSET,
            bitwise_chip.clone(),
            tester.range_checker(),
        );
        chip.set_trace_height(MAX_INS_CAPACITY);

        assert_eq!(
            chip.expr().builder.num_variables,
            2,
            "Fp2MulDiv should only introduce new z Fp2 variable (2 Fp var)"
        );

        let num_ops = 10;
        for _ in 0..num_ops {
            set_and_execute_rand(&mut tester, &mut chip, &modulus);
        }

        let tester = tester.build().load(chip).load(bitwise_chip).finalize();
        tester.simple_test().expect("Verification failed");
    }
}
