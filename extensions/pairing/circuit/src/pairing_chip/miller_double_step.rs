use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use openvm_algebra_circuit::Fp2;
use openvm_circuit::{arch::VmChipWrapper, system::memory::OfflineMemory};
use openvm_circuit_derive::InstructionExecutor;
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerBus,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreChip,
};
use openvm_pairing_transpiler::PairingOpcode;
use openvm_rv32_adapters::Rv32VecHeapAdapterChip;
use openvm_stark_backend::p3_field::PrimeField32;

// Input: AffinePoint<Fp2>: 4 field elements
// Output: (AffinePoint<Fp2>, Fp2, Fp2) -> 8 field elements
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct MillerDoubleStepChip<
    F: PrimeField32,
    const INPUT_BLOCKS: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 1, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<
        F: PrimeField32,
        const INPUT_BLOCKS: usize,
        const OUTPUT_BLOCKS: usize,
        const BLOCK_SIZE: usize,
    > MillerDoubleStepChip<F, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 1, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        config: ExprBuilderConfig,
        offset: usize,
        range_checker: SharedVariableRangeCheckerChip,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    ) -> Self {
        let expr = miller_double_step_expr(config, range_checker.bus());
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![PairingOpcode::MILLER_DOUBLE_STEP as usize],
            vec![],
            range_checker,
            "MillerDoubleStep",
            false,
        );
        Self(VmChipWrapper::new(adapter, core, offline_memory))
    }
}

// Ref: https://github.com/openvm-org/openvm/blob/f7d6fa7b8ef247e579740eb652fcdf5a04259c28/lib/ecc-execution/src/common/miller_step.rs#L7
pub fn miller_double_step_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x_s = Fp2::new(builder.clone());
    let mut y_s = Fp2::new(builder.clone());

    let mut three_x_square = x_s.square().int_mul([3, 0]);
    let mut lambda = three_x_square.div(&mut y_s.int_mul([2, 0]));
    let mut x_2s = lambda.square().sub(&mut x_s.int_mul([2, 0]));
    let mut y_2s = lambda.mul(&mut (x_s.sub(&mut x_2s))).sub(&mut y_s);
    x_2s.save_output();
    y_2s.save_output();

    let mut b = lambda.neg();
    let mut c = lambda.mul(&mut x_s).sub(&mut y_s);
    b.save_output();
    c.save_output();

    let builder = builder.borrow().clone();
    FieldExpr::new(builder, range_bus, false)
}

#[cfg(test)]
mod tests {
    use openvm_circuit::arch::testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS};
    use openvm_circuit_primitives::bitwise_op_lookup::{
        BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
    };
    use openvm_ecc_guest::AffinePoint;
    use openvm_instructions::{riscv::RV32_CELL_BITS, LocalOpcode};
    use openvm_mod_circuit_builder::test_utils::{
        biguint_to_limbs, bls12381_fq_to_biguint, bn254_fq_to_biguint,
    };
    use openvm_pairing_guest::{
        bls12_381::{BLS12_381_LIMB_BITS, BLS12_381_MODULUS, BLS12_381_NUM_LIMBS},
        bn254::{BN254_LIMB_BITS, BN254_MODULUS, BN254_NUM_LIMBS},
        halo2curves_shims::{bls12_381::Bls12_381, bn254::Bn254},
        pairing::MillerStep,
    };
    use openvm_pairing_transpiler::PairingOpcode;
    use openvm_rv32_adapters::{rv32_write_heap_default, Rv32VecHeapAdapterChip};
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    type F = BabyBear;

    #[test]
    #[allow(non_snake_case)]
    fn test_miller_double_bn254() {
        use halo2curves_axiom::bn256::G2Affine;
        const NUM_LIMBS: usize = 32;
        const LIMB_BITS: usize = 8;
        const BLOCK_SIZE: usize = 32;

        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: BN254_MODULUS.clone(),
            limb_bits: BN254_LIMB_BITS,
            num_limbs: BN254_NUM_LIMBS,
        };
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
        let adapter = Rv32VecHeapAdapterChip::<F, 1, 4, 8, BLOCK_SIZE, BLOCK_SIZE>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_bridge(),
            tester.address_bits(),
            bitwise_chip.clone(),
        );
        let mut chip = MillerDoubleStepChip::new(
            adapter,
            config,
            PairingOpcode::CLASS_OFFSET,
            tester.range_checker(),
            tester.offline_memory_mutex_arc(),
        );

        let mut rng0 = StdRng::seed_from_u64(2);
        let Q = G2Affine::random(&mut rng0);
        let inputs = [Q.x.c0, Q.x.c1, Q.y.c0, Q.y.c1].map(bn254_fq_to_biguint);

        let Q_ecpoint = AffinePoint { x: Q.x, y: Q.y };
        let (Q_acc_init, l_init) = Bn254::miller_double_step(&Q_ecpoint);
        let result = chip
            .0
            .core
            .expr()
            .execute_with_output(inputs.to_vec(), vec![]);
        assert_eq!(result.len(), 8); // AffinePoint<Fp2> and two Fp2 coefficients
        assert_eq!(result[0], bn254_fq_to_biguint(Q_acc_init.x.c0));
        assert_eq!(result[1], bn254_fq_to_biguint(Q_acc_init.x.c1));
        assert_eq!(result[2], bn254_fq_to_biguint(Q_acc_init.y.c0));
        assert_eq!(result[3], bn254_fq_to_biguint(Q_acc_init.y.c1));
        assert_eq!(result[4], bn254_fq_to_biguint(l_init.b.c0));
        assert_eq!(result[5], bn254_fq_to_biguint(l_init.b.c1));
        assert_eq!(result[6], bn254_fq_to_biguint(l_init.c.c0));
        assert_eq!(result[7], bn254_fq_to_biguint(l_init.c.c1));

        let input_limbs = inputs
            .map(|x| biguint_to_limbs::<NUM_LIMBS>(x, LIMB_BITS).map(BabyBear::from_canonical_u32));

        let instruction = rv32_write_heap_default(
            &mut tester,
            input_limbs.to_vec(),
            vec![],
            chip.0.core.air.offset + PairingOpcode::MILLER_DOUBLE_STEP as usize,
        );

        tester.execute(&mut chip, &instruction);
        let tester = tester.build().load(chip).load(bitwise_chip).finalize();
        tester.simple_test().expect("Verification failed");
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_miller_double_bls12_381() {
        use halo2curves_axiom::bls12_381::G2Affine;
        const NUM_LIMBS: usize = 48;
        const LIMB_BITS: usize = 8;
        const BLOCK_SIZE: usize = 16;

        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: BLS12_381_MODULUS.clone(),
            limb_bits: BLS12_381_LIMB_BITS,
            num_limbs: BLS12_381_NUM_LIMBS,
        };
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
        let adapter = Rv32VecHeapAdapterChip::<F, 1, 12, 24, BLOCK_SIZE, BLOCK_SIZE>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_bridge(),
            tester.address_bits(),
            bitwise_chip.clone(),
        );
        let mut chip = MillerDoubleStepChip::new(
            adapter,
            config,
            PairingOpcode::CLASS_OFFSET,
            tester.range_checker(),
            tester.offline_memory_mutex_arc(),
        );

        let mut rng0 = StdRng::seed_from_u64(12);
        let Q = G2Affine::random(&mut rng0);
        let inputs = [Q.x.c0, Q.x.c1, Q.y.c0, Q.y.c1].map(bls12381_fq_to_biguint);

        let Q_ecpoint = AffinePoint { x: Q.x, y: Q.y };
        let (Q_acc_init, l_init) = Bls12_381::miller_double_step(&Q_ecpoint);
        let result = chip
            .0
            .core
            .expr()
            .execute_with_output(inputs.to_vec(), vec![]);
        assert_eq!(result.len(), 8); // AffinePoint<Fp2> and two Fp2 coefficients
        assert_eq!(result[0], bls12381_fq_to_biguint(Q_acc_init.x.c0));
        assert_eq!(result[1], bls12381_fq_to_biguint(Q_acc_init.x.c1));
        assert_eq!(result[2], bls12381_fq_to_biguint(Q_acc_init.y.c0));
        assert_eq!(result[3], bls12381_fq_to_biguint(Q_acc_init.y.c1));
        assert_eq!(result[4], bls12381_fq_to_biguint(l_init.b.c0));
        assert_eq!(result[5], bls12381_fq_to_biguint(l_init.b.c1));
        assert_eq!(result[6], bls12381_fq_to_biguint(l_init.c.c0));
        assert_eq!(result[7], bls12381_fq_to_biguint(l_init.c.c1));

        let input_limbs = inputs
            .map(|x| biguint_to_limbs::<NUM_LIMBS>(x, LIMB_BITS).map(BabyBear::from_canonical_u32));

        let instruction = rv32_write_heap_default(
            &mut tester,
            input_limbs.to_vec(),
            vec![],
            chip.0.core.air.offset + PairingOpcode::MILLER_DOUBLE_STEP as usize,
        );

        tester.execute(&mut chip, &instruction);
        let tester = tester.build().load(chip).load(bitwise_chip).finalize();
        tester.simple_test().expect("Verification failed");
    }
}
