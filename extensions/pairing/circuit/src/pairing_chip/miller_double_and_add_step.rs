use std::{cell::RefCell, rc::Rc};

use openvm_algebra_circuit::Fp2;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_derive::{InsExecutorE1, InstructionExecutor};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir,
};
use openvm_pairing_transpiler::PairingOpcode;
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterStep};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{PairingHeapAdapterAir, PairingHeapAdapterStep};

use super::PairingHeapAdapterChip;

// Input: two AffinePoint<Fp2>: 4 field elements each
// Output: (AffinePoint<Fp2>, UnevaluatedLine<Fp2>, UnevaluatedLine<Fp2>) -> 2*2 + 2*2 + 2*2 = 12
// field elements
#[derive(Chip, ChipUsageGetter, InstructionExecutor, InsExecutorE1)]
pub struct MillerDoubleAndAddStepChip<
    F: PrimeField32,
    const INPUT_BLOCKS: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(pub PairingHeapAdapterChip<F, 2,INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE>);

impl<
        F: PrimeField32,
        const INPUT_BLOCKS: usize,
        const OUTPUT_BLOCKS: usize,
        const BLOCK_SIZE: usize,
    > MillerDoubleAndAddStepChip<F, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        execution_bridge: ExecutionBridge,
        memory_bridge: MemoryBridge,
        mem_helper: SharedMemoryHelper<F>,
        pointer_max_bits: usize,
        config: ExprBuilderConfig,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        range_checker: SharedVariableRangeCheckerChip,
        height: usize,
    ) -> Self {
        let expr = miller_double_and_add_step_expr(config, range_checker.bus());
        let local_opcode_idx = vec![PairingOpcode::MILLER_DOUBLE_AND_ADD_STEP as usize];

        let air = PairingHeapAdapterAir::new(
            Rv32VecHeapAdapterAir::new(
                execution_bridge,
                memory_bridge,
                bitwise_lookup_chip.bus(),
                pointer_max_bits,
            ),
            FieldExpressionCoreAir::new(expr.clone(), offset, local_opcode_idx.clone(), vec![]),
        );

        let step = PairingHeapAdapterStep::new(
            Rv32VecHeapAdapterStep::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            offset,
            local_opcode_idx,
            vec![],
            range_checker,
            "MillerDoubleAndAddStep",
            false,
        );
        Self(PairingHeapAdapterChip::new(air, step, height, mem_helper))
    }
}

// Ref: openvm_pairing_guest::miller_step
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
    FieldExpr::new(builder, range_bus, false)
}

#[cfg(test)]
mod tests {
    use halo2curves_axiom::bn256::G2Affine;
    use openvm_circuit::arch::testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS};
    use openvm_circuit_primitives::bitwise_op_lookup::{
        BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
    };
    use openvm_ecc_guest::AffinePoint;
    use openvm_instructions::{riscv::RV32_CELL_BITS, LocalOpcode};
    use openvm_mod_circuit_builder::test_utils::{biguint_to_limbs, bn254_fq_to_biguint};
    use openvm_pairing_guest::{
        bn254::BN254_MODULUS, halo2curves_shims::bn254::Bn254, pairing::MillerStep,
    };
    use openvm_pairing_transpiler::PairingOpcode;
    use openvm_rv32_adapters::rv32_write_heap_default;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    type F = BabyBear;
    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    const BLOCK_SIZE: usize = 32;
    const MAX_INS_CAPACITY: usize = 128;

    #[test]
    #[allow(non_snake_case)]
    fn test_miller_double_and_add() {
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: BN254_MODULUS.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
        let mut chip = MillerDoubleAndAddStepChip::<F, 4, 12, BLOCK_SIZE>::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            tester.memory_helper(),
            tester.address_bits(),
            config,
            PairingOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
            tester.range_checker(),
            MAX_INS_CAPACITY,
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
            .step
            .expr
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
            chip.0.step.offset + PairingOpcode::MILLER_DOUBLE_AND_ADD_STEP as usize,
        );

        tester.execute(&mut chip, &instruction);
        let tester = tester.build().load(chip).load(bitwise_chip).finalize();
        tester.simple_test().expect("Verification failed");
    }
}
