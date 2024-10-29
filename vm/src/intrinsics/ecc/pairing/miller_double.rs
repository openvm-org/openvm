use std::{cell::RefCell, rc::Rc};

use ax_circuit_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerBus,
};
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, FieldExpr},
    field_extension::Fp2,
};
use num_bigint_dig::BigUint;

use super::super::FIELD_ELEMENT_BITS;

pub fn miller_double_expr(
    modulus: BigUint,
    num_limbs: usize,
    limb_bits: usize,
    range_bus: VariableRangeCheckerBus,
) -> FieldExpr {
    assert!(modulus.bits() <= num_limbs * limb_bits);
    let subair = CheckCarryModToZeroSubAir::new(
        modulus.clone(),
        limb_bits,
        range_bus.index,
        range_bus.range_max_bits,
        FIELD_ELEMENT_BITS,
    );
    let builder = ExprBuilder::new(modulus, limb_bits, num_limbs, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut s_x = Fp2::new(builder.clone());
    let mut s_y = Fp2::new(builder.clone());

    let mut three_x_square = s_x.square().int_mul([3, 0]);
    let mut lambda = three_x_square.div(&mut s_y.int_mul([2, 0]));
    let mut x_2s = lambda.square().sub(&mut s_x.int_mul([2, 0]));
    let mut y_2s = lambda.mul(&mut (s_x.sub(&mut x_2s))).sub(&mut s_y);
    x_2s.save_output();
    y_2s.save_output();

    let mut b = lambda.int_mul([-1, 0]);
    let mut c = lambda.mul(&mut s_x).sub(&mut s_y);
    b.save_output();
    c.save_output();

    let builder = builder.borrow().clone();
    FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus,
    }
}

#[cfg(test)]
mod tests {
    use ax_ecc_execution::common::{miller_double_step, EcPoint};
    use axvm_ecc_constants::BN254;
    use axvm_instructions::UsizeOpcode;
    // use halo2curves_axiom::bls12_381::{Fq, Fq2, G2Affine};
    use halo2curves_axiom::bn256::{Fq, Fq2, G2Affine};
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        arch::{instructions::PairingOpcode, testing::VmChipTestBuilder, VmChipWrapper},
        intrinsics::field_expression::FieldExpressionCoreChip,
        rv32im::adapters::Rv32VecHeapAdapterChip,
        utils::{biguint_to_limbs, rv32_write_heap_default},
    };

    // Only for testing, not the most performant
    fn fq_to_biguint(fq: Fq) -> BigUint {
        BigUint::from_bytes_le(&fq.to_bytes())
    }

    type F = BabyBear;
    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    #[test]
    #[allow(non_snake_case)]
    fn test_miller_double() {
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let expr = miller_double_expr(
            BN254.MODULUS.clone(),
            NUM_LIMBS,
            LIMB_BITS,
            tester.memory_controller().borrow().range_checker.bus(),
        );
        let core = FieldExpressionCoreChip::new(
            expr,
            PairingOpcode::default_offset(),
            vec![PairingOpcode::MILLER_DOUBLE as usize],
            tester.memory_controller().borrow().range_checker.clone(),
            "MillerDouble",
        );
        let adapter = Rv32VecHeapAdapterChip::<F, 1, 4, 8, NUM_LIMBS, NUM_LIMBS>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        );
        let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

        let mut rng0 = StdRng::seed_from_u64(2);
        let Q = G2Affine::random(&mut rng0);
        let inputs = [Q.x.c0, Q.x.c1, Q.y.c0, Q.y.c1].map(fq_to_biguint);
        println!(
            "inputs: {} {} {} {}",
            inputs[0], inputs[1], inputs[2], inputs[3]
        );

        let Q_ecpoint = EcPoint { x: Q.x, y: Q.y };
        let (Q_acc_init, l_init) = miller_double_step::<Fq, Fq2>(Q_ecpoint.clone());
        let result = chip
            .core
            .expr()
            .execute_with_output(inputs.to_vec(), vec![]);
        assert_eq!(result.len(), 8); // EcPoint<Fp2> and two Fp2 coefficients
        println!("{} v.s. {}", result[0], fq_to_biguint(Q_acc_init.x.c0));
        println!("{} v.s. {}", result[1], fq_to_biguint(Q_acc_init.x.c1));
        println!("{} v.s. {}", result[2], fq_to_biguint(Q_acc_init.y.c0));
        println!("{} v.s. {}", result[3], fq_to_biguint(Q_acc_init.y.c1));
        println!("{} v.s. {}", result[4], fq_to_biguint(l_init.b.c0));
        println!("{} v.s. {}", result[5], fq_to_biguint(l_init.b.c1));
        println!("{} v.s. {}", result[6], fq_to_biguint(l_init.c.c0));
        println!("{} v.s. {}", result[7], fq_to_biguint(l_init.c.c1));

        assert_eq!(result[0], fq_to_biguint(Q_acc_init.x.c0));
        assert_eq!(result[1], fq_to_biguint(Q_acc_init.x.c1));
        assert_eq!(result[2], fq_to_biguint(Q_acc_init.y.c0));
        assert_eq!(result[3], fq_to_biguint(Q_acc_init.y.c1));
        assert_eq!(result[4], fq_to_biguint(l_init.b.c0));
        assert_eq!(result[5], fq_to_biguint(l_init.b.c1));
        assert_eq!(result[6], fq_to_biguint(l_init.c.c0));
        assert_eq!(result[7], fq_to_biguint(l_init.c.c1));

        let input_limbs = inputs
            .map(|x| biguint_to_limbs::<NUM_LIMBS>(x, LIMB_BITS).map(BabyBear::from_canonical_u32));

        let instruction = rv32_write_heap_default(
            &mut tester,
            input_limbs.to_vec(),
            vec![],
            chip.core.air.offset + PairingOpcode::MILLER_DOUBLE as usize,
        );

        tester.execute(&mut chip, instruction);
        let tester = tester.build().load(chip).finalize();
        tester.simple_test().expect("Verification failed");
    }
}
