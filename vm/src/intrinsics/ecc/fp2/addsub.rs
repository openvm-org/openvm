use std::{cell::RefCell, rc::Rc};

use ax_circuit_primitives::var_range::VariableRangeCheckerBus;
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, ExprBuilderConfig, FieldExpr},
    field_extension::Fp2,
};

pub fn fp2_addsub_expr(config: ExprBuilderConfig, range_bus: VariableRangeCheckerBus) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x = Fp2::new(builder.clone());
    let mut y = Fp2::new(builder.clone());
    let add = x.add(&mut y);
    let sub = x.sub(&mut y);

    let flag = builder.borrow_mut().new_flag();
    let mut z = Fp2::select(flag, &add, &sub);
    z.save_output();

    let builder = builder.borrow().clone();
    FieldExpr::new(builder, range_bus)
}

#[cfg(test)]
mod tests {
    use ax_ecc_primitives::{
        field_expression::ExprBuilderConfig,
        test_utils::{bn254_fq2_to_biguint_vec, bn254_fq_to_biguint},
    };
    use axvm_ecc_constants::BN254;
    use axvm_instructions::UsizeOpcode;
    use halo2curves_axiom::{
        bn256::{Fq, Fq2, G2Affine},
        ff::Field,
    };
    use num_bigint_dig::BigUint;
    use num_traits::FromPrimitive;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use rand::{rngs::StdRng, SeedableRng};

    use super::fp2_addsub_expr;
    use crate::{
        arch::{instructions::EccOpcode, testing::VmChipTestBuilder, VmChipWrapper},
        intrinsics::field_expression::FieldExpressionCoreChip,
        rv32im::adapters::Rv32VecHeapAdapterChip,
        utils::{biguint_to_limbs, rv32_write_heap_default},
    };

    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    type F = BabyBear;

    #[test]
    fn test_fp2_addsub() {
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: BN254.MODULUS.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let expr = fp2_addsub_expr(
            config,
            tester.memory_controller().borrow().range_checker.bus(),
        );

        let core = FieldExpressionCoreChip::new(
            expr,
            EccOpcode::default_offset(),
            vec![EccOpcode::EC_ADD_NE as usize],
            tester.memory_controller().borrow().range_checker.clone(),
            "Fp2AddSub",
        );
        let adapter = Rv32VecHeapAdapterChip::<F, 2, 1, 1, NUM_LIMBS, NUM_LIMBS>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        );
        let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

        let mut rng = StdRng::seed_from_u64(42);
        let x = Fq2::random(&mut rng);
        let y = Fq2::random(&mut rng);
        let expected_sum = bn254_fq2_to_biguint_vec(&(x + y));
        let inputs = [x.c0, x.c1, y.c0, y.c1].map(|x| bn254_fq_to_biguint(&x));
        let r = chip
            .core
            .air
            .expr
            .execute_with_output(inputs.to_vec(), vec![true]);

        assert_eq!(r.len(), 2);
        assert_eq!(r[0], expected_sum[0]);
        assert_eq!(r[1], expected_sum[1]);
    }
}
