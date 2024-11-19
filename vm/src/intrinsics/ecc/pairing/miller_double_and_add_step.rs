use std::{cell::RefCell, rc::Rc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::var_range::VariableRangeCheckerBus;
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, ExprBuilderConfig, FieldExpr},
    field_extension::Fp2,
};
use axvm_circuit_derive::InstructionExecutor;
use p3_field::PrimeField32;

use crate::{
    arch::{instructions::PairingOpcode, VmChipWrapper},
    intrinsics::field_expression::FieldExpressionCoreChip,
    rv32im::adapters::Rv32VecHeapAdapterChip,
    system::memory::MemoryControllerRef,
};

// Input: two AffinePoint<Fp2>: 4 field elements each
// Output: (AffinePoint<Fp2>, UnevaluatedLine<Fp2>, UnevaluatedLine<Fp2>) -> 2*2 + 2*2 + 2*2 = 12 field elements
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct MillerDoubleAndAddStepChip<
    F: PrimeField32,
    const INPUT_BLOCKS: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    pub  VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 2, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<
        F: PrimeField32,
        const INPUT_BLOCKS: usize,
        const OUTPUT_BLOCKS: usize,
        const BLOCK_SIZE: usize,
    > MillerDoubleAndAddStepChip<F, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        memory_controller: MemoryControllerRef<F>,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> Self {
        let expr =
            miller_double_and_add_step_expr(config, memory_controller.borrow().range_checker.bus());
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![PairingOpcode::MILLER_DOUBLE_AND_ADD_STEP as usize],
            vec![],
            memory_controller.borrow().range_checker.clone(),
            "MillerDoubleAndAddStep",
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}

// Ref: https://github.com/axiom-crypto/afs-prototype/blob/043968891d596acdc82c8e3a9126a555fe178d43/lib/ecc-execution/src/common/miller_step.rs#L72
pub fn miller_double_and_add_step_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x_s = Fp2::new(builder.clone());
    let mut y_s = Fp2::new(builder.clone());
    let mut x_q = Fp2::new(builder.clone());
    let mut y_q = Fp2::new(builder.clone());

    // λ1 = (y_s - y_q) / (x_s - x_q)
    let mut lambda1 = y_s.sub(&mut y_q).div(&mut x_s.sub(&mut x_q));
    let mut x_sq = lambda1.square().sub(&mut x_s).sub(&mut x_q);
    // λ2 = -λ1 - 2y_s / (x_{s+q} - x_s)
    let mut lambda2 = lambda1
        .neg()
        .sub(&mut y_s.int_mul([2, 0]).div(&mut x_sq.sub(&mut x_s)));
    let mut x_sqs = lambda2.square().sub(&mut x_s).sub(&mut x_sq);
    let mut y_sqs = lambda2.mul(&mut (x_s.sub(&mut x_sqs))).sub(&mut y_s);

    x_sqs.save_output();
    y_sqs.save_output();

    let mut b0 = lambda1.neg();
    let mut c0 = lambda1.mul(&mut x_s).sub(&mut y_s);
    b0.save_output();
    c0.save_output();

    let mut b1 = lambda2.neg();
    let mut c1 = lambda2.mul(&mut x_s).sub(&mut y_s);
    b1.save_output();
    c1.save_output();

    let builder = builder.borrow().clone();
    FieldExpr::new(builder, range_bus)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ax_circuit_primitives::bitwise_op_lookup::{
        BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    };
    use ax_ecc_execution::curves::bn254::Bn254;
    use ax_ecc_primitives::test_utils::bn254_fq_to_biguint;
    use axvm_ecc::{pairing::MillerStep, AffinePoint};
    use axvm_ecc_constants::BN254;
    use axvm_instructions::{riscv::RV32_CELL_BITS, UsizeOpcode};
    use halo2curves_axiom::bn256::G2Affine;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        arch::{instructions::PairingOpcode, testing::VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        rv32im::adapters::Rv32VecHeapAdapterChip,
        utils::{biguint_to_limbs, rv32_write_heap_default},
    };

    type F = BabyBear;
    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    const BLOCK_SIZE: usize = 32;

    #[test]
    #[allow(non_snake_case)]
    fn test_miller_double_and_add() {
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));
        let adapter = Rv32VecHeapAdapterChip::<F, 2, 4, 12, BLOCK_SIZE, BLOCK_SIZE>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
            bitwise_chip.clone(),
        );
        let mut chip = MillerDoubleAndAddStepChip::new(
            adapter,
            tester.memory_controller(),
            ExprBuilderConfig {
                modulus: BN254.MODULUS.clone(),
                limb_bits: LIMB_BITS,
                num_limbs: NUM_LIMBS,
            },
            PairingOpcode::default_offset(),
        );

        let mut rng0 = StdRng::seed_from_u64(2);
        let Q = G2Affine::random(&mut rng0);
        let Q2 = G2Affine::random(&mut rng0);
        let inputs = [
            Q.x.c0, Q.x.c1, Q.y.c0, Q.y.c1, Q2.x.c0, Q2.x.c1, Q2.y.c0, Q2.y.c1,
        ]
        .map(bn254_fq_to_biguint);

        let Q_ecpoint = AffinePoint { x: Q.x, y: Q.y };
        let Q_ecpoint2 = AffinePoint { x: Q2.x, y: Q2.y };
        let (Q_daa, l_qa, l_sqs) = Bn254::miller_double_and_add_step(&Q_ecpoint, &Q_ecpoint2);
        let result = chip
            .0
            .core
            .expr()
            .execute_with_output(inputs.to_vec(), vec![]);
        assert_eq!(result.len(), 12); // AffinePoint<Fp2> and 4 Fp2 coefficients
        assert_eq!(result[0], bn254_fq_to_biguint(Q_daa.x.c0));
        assert_eq!(result[1], bn254_fq_to_biguint(Q_daa.x.c1));
        assert_eq!(result[2], bn254_fq_to_biguint(Q_daa.y.c0));
        assert_eq!(result[3], bn254_fq_to_biguint(Q_daa.y.c1));
        assert_eq!(result[4], bn254_fq_to_biguint(l_qa.b.c0));
        assert_eq!(result[5], bn254_fq_to_biguint(l_qa.b.c1));
        assert_eq!(result[6], bn254_fq_to_biguint(l_qa.c.c0));
        assert_eq!(result[7], bn254_fq_to_biguint(l_qa.c.c1));
        assert_eq!(result[8], bn254_fq_to_biguint(l_sqs.b.c0));
        assert_eq!(result[9], bn254_fq_to_biguint(l_sqs.b.c1));
        assert_eq!(result[10], bn254_fq_to_biguint(l_sqs.c.c0));
        assert_eq!(result[11], bn254_fq_to_biguint(l_sqs.c.c1));

        let input1_limbs = inputs[0..4]
            .iter()
            .map(|x| {
                biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS)
                    .map(BabyBear::from_canonical_u32)
            })
            .collect::<Vec<_>>();

        let input2_limbs = inputs[4..8]
            .iter()
            .map(|x| {
                biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS)
                    .map(BabyBear::from_canonical_u32)
            })
            .collect::<Vec<_>>();

        let instruction = rv32_write_heap_default(
            &mut tester,
            input1_limbs,
            input2_limbs,
            chip.0.core.air.offset + PairingOpcode::MILLER_DOUBLE_AND_ADD_STEP as usize,
        );

        tester.execute(&mut chip, instruction);
        let tester = tester.build().load(chip).load(bitwise_chip).finalize();
        tester.simple_test().expect("Verification failed");
    }
}
