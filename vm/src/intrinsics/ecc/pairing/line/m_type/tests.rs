use ax_ecc_execution::{
    common::{EcPoint, Fp2Constructor},
    curves::bls12_381::point_to_023,
};
use ax_ecc_primitives::{
    field_expression::ExprBuilderConfig,
    test_utils::{
        bls12381_fq12_to_biguint_vec, bls12381_fq2_to_biguint_vec, bls12381_fq_to_biguint,
    },
};
use axvm_ecc_constants::BLS12381;
use axvm_instructions::{PairingOpcode, UsizeOpcode};
use halo2curves_axiom::{
    bls12_381::{Fq, Fq12, Fq2, G1Affine},
    ff::Field,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    arch::{testing::VmChipTestBuilder, VmChipWrapper},
    intrinsics::{
        ecc::pairing::{mul_023_by_023_expr, mul_by_02345_expr},
        field_expression::FieldExpressionCoreChip,
    },
    rv32im::adapters::Rv32VecHeapAdapterChip,
    utils::{biguint_to_limbs, rv32_write_heap_default},
};

type F = BabyBear;

#[test]
fn test_mul_023_by_023() {
    const NUM_LIMBS: usize = 64;
    const LIMB_BITS: usize = 8;

    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let expr = mul_023_by_023_expr(
        ExprBuilderConfig {
            modulus: BLS12381.MODULUS.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        },
        tester.memory_controller().borrow().range_checker.bus(),
        BLS12381.XI,
    );
    let core = FieldExpressionCoreChip::new(
        expr,
        PairingOpcode::default_offset(),
        vec![PairingOpcode::MUL_023_BY_023 as usize],
        vec![],
        tester.memory_controller().borrow().range_checker.clone(),
        "Mul023By023",
    );
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 4, 10, NUM_LIMBS, NUM_LIMBS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );

    let mut rng0 = StdRng::seed_from_u64(15);
    let mut rng1 = StdRng::seed_from_u64(95);
    let rnd_pt_0 = G1Affine::random(&mut rng0);
    let rnd_pt_1 = G1Affine::random(&mut rng1);
    let ec_pt_0 = EcPoint::<Fq> {
        x: rnd_pt_0.x,
        y: rnd_pt_0.y,
    };
    let ec_pt_1 = EcPoint::<Fq> {
        x: rnd_pt_1.x,
        y: rnd_pt_1.y,
    };
    let line0 = point_to_023::<Fq, Fq2>(ec_pt_0);
    let line1 = point_to_023::<Fq, Fq2>(ec_pt_1);
    let input_line0 = [
        bls12381_fq2_to_biguint_vec(line0.b),
        bls12381_fq2_to_biguint_vec(line0.c),
    ]
    .concat();
    let input_line1 = [
        bls12381_fq2_to_biguint_vec(line1.b),
        bls12381_fq2_to_biguint_vec(line1.c),
    ]
    .concat();

    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

    let vars = chip
        .core
        .air
        .expr
        .execute([input_line0.clone(), input_line1.clone()].concat(), vec![]);
    let output_indices = chip.core.air.expr.builder.output_indices.clone();
    let output = output_indices
        .iter()
        .map(|i| vars[*i].clone())
        .collect::<Vec<_>>();
    assert_eq!(output.len(), 10);

    let r_cmp = ax_ecc_execution::curves::bls12_381::mul_023_by_023::<Fq, Fq2>(
        line0,
        line1,
        Fq2::new(Fq::one(), Fq::one()),
    );
    let r_cmp_bigint = r_cmp
        .map(|x| [bls12381_fq_to_biguint(x.c0), bls12381_fq_to_biguint(x.c1)])
        .concat();

    for i in 0..10 {
        if i >= 2 {
            // Skip c1 in 02345 representation
            assert_eq!(output[i], r_cmp_bigint[i + 2]);
        } else {
            assert_eq!(output[i], r_cmp_bigint[i]);
        }
    }

    let input_line0_limbs = input_line0
        .iter()
        .map(|x| {
            biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32)
        })
        .collect::<Vec<_>>();
    let input_line1_limbs = input_line1
        .iter()
        .map(|x| {
            biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32)
        })
        .collect::<Vec<_>>();

    let instruction = rv32_write_heap_default(
        &mut tester,
        input_line0_limbs,
        input_line1_limbs,
        chip.core.air.offset + PairingOpcode::MUL_023_BY_023 as usize,
    );

    tester.execute(&mut chip, instruction);
    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}

// NOTE[yj]: This test requires RUST_MIN_STACK=8388608 to run without overflowing the stack, so it is ignored by the test runner for now
#[test]
#[ignore]
fn test_mul_by_02345() {
    const NUM_LIMBS: usize = 64;
    const LIMB_BITS: usize = 8;

    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let expr = mul_by_02345_expr(
        ExprBuilderConfig {
            modulus: BLS12381.MODULUS.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        },
        tester.memory_controller().borrow().range_checker.bus(),
        BLS12381.XI,
    );
    let core = FieldExpressionCoreChip::new(
        expr,
        PairingOpcode::default_offset(),
        vec![PairingOpcode::MUL_BY_02345 as usize],
        vec![],
        tester.memory_controller().borrow().range_checker.clone(),
        "MulBy02345",
    );
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 12, 12, NUM_LIMBS, NUM_LIMBS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );

    let mut rng = StdRng::seed_from_u64(8);
    let f = Fq12::random(&mut rng);
    let mut rng = StdRng::seed_from_u64(12);
    let x0 = Fq2::random(&mut rng);
    let mut rng = StdRng::seed_from_u64(5);
    let x2 = Fq2::random(&mut rng);
    let mut rng = StdRng::seed_from_u64(77);
    let x3 = Fq2::random(&mut rng);
    let mut rng = StdRng::seed_from_u64(31);
    let x4 = Fq2::random(&mut rng);
    let mut rng = StdRng::seed_from_u64(1);
    let x5 = Fq2::random(&mut rng);

    let input_f = bls12381_fq12_to_biguint_vec(f);
    let input_x = [
        bls12381_fq2_to_biguint_vec(x0),
        bls12381_fq2_to_biguint_vec(Fq2::zero()),
        bls12381_fq2_to_biguint_vec(x2),
        bls12381_fq2_to_biguint_vec(x3),
        bls12381_fq2_to_biguint_vec(x4),
        bls12381_fq2_to_biguint_vec(x5),
    ]
    .concat();

    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

    let vars = chip
        .core
        .air
        .expr
        .execute([input_f.clone(), input_x.clone()].concat(), vec![]);
    let output_indices = chip.core.air.expr.builder.output_indices.clone();
    let output = output_indices
        .iter()
        .map(|i| vars[*i].clone())
        .collect::<Vec<_>>();
    assert_eq!(output.len(), 12);

    let r_cmp = ax_ecc_execution::curves::bls12_381::mul_by_02345::<Fq, Fq2, Fq12>(
        f,
        [x0, Fq2::ZERO, x2, x3, x4, x5],
    );
    let r_cmp_bigint = bls12381_fq12_to_biguint_vec(r_cmp);

    for i in 0..12 {
        assert_eq!(output[i], r_cmp_bigint[i]);
    }

    let input_f_limbs = input_f
        .iter()
        .map(|x| {
            biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32)
        })
        .collect::<Vec<_>>();
    let input_x_limbs = input_x
        .iter()
        .map(|x| {
            biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32)
        })
        .collect::<Vec<_>>();

    let instruction = rv32_write_heap_default(
        &mut tester,
        input_f_limbs,
        input_x_limbs,
        chip.core.air.offset + PairingOpcode::MUL_BY_02345 as usize,
    );

    tester.execute(&mut chip, instruction);
    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}
