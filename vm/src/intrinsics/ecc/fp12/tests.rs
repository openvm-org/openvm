use ax_ecc_execution::{common::EcPoint, curves::bn254::point_to_013};
use ax_ecc_primitives::{
    field_expression::FieldExpr,
    test_utils::{
        bls12381_fq12_random, bn254_fq12_random, bn254_fq12_to_biguint_vec,
        bn254_fq2_to_biguint_vec, bn254_fq_to_biguint,
    },
};
use axvm_ecc_constants::{BLS12381, BN254};
use axvm_instructions::{Bls12381Fp12Opcode, Bn254Fp12Opcode, Fp12Opcode, UsizeOpcode};
use halo2curves_axiom::bn256::{Fq, Fq2, G1Affine};
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::{rngs::StdRng, SeedableRng};

use super::{fp12_add_expr, fp12_mul_expr, fp12_sub_expr};
use crate::{
    arch::{testing::VmChipTestBuilder, VmChipWrapper},
    intrinsics::{ecc::fp12::mul_013_by_013_expr, field_expression::FieldExpressionCoreChip},
    rv32im::adapters::Rv32VecHeapAdapterChip,
    utils::{biguint_to_limbs, rv32_write_heap_default},
};

const BN254_NUM_LIMBS: usize = 32;
const BN254_LIMB_BITS: usize = 8;

const BLS12381_NUM_LIMBS: usize = 64;
const BLS12381_LIMB_BITS: usize = 8;

type F = BabyBear;

#[allow(clippy::too_many_arguments)]
fn test_fp12_fn<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    mut tester: VmChipTestBuilder<F>,
    expr: FieldExpr,
    offset: usize,
    local_opcode_idx: usize,
    name: &str,
    x: Vec<BigUint>,
    y: Vec<BigUint>,
    var_len: usize,
) {
    let core = FieldExpressionCoreChip::new(
        expr,
        offset,
        vec![local_opcode_idx],
        tester.memory_controller().borrow().range_checker.clone(),
        name,
    );
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 12, 12, NUM_LIMBS, NUM_LIMBS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );

    let x_limbs = x
        .iter()
        .map(|x| {
            biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32)
        })
        .collect::<Vec<[BabyBear; NUM_LIMBS]>>();
    let y_limbs = y
        .iter()
        .map(|y| {
            biguint_to_limbs::<NUM_LIMBS>(y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32)
        })
        .collect::<Vec<[BabyBear; NUM_LIMBS]>>();
    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

    let res = chip.core.air.expr.execute([x, y].concat(), vec![]);
    assert_eq!(res.len(), var_len);

    let instruction = rv32_write_heap_default(
        &mut tester,
        x_limbs,
        y_limbs,
        chip.core.air.offset + local_opcode_idx,
    );
    tester.execute(&mut chip, instruction);

    let run_tester = tester.build().load(chip).finalize();
    run_tester.simple_test().expect("Verification failed");
}

#[test]
fn test_fp12_sub_bn254() {
    let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = BN254.MODULUS.clone();
    let expr = fp12_sub_expr(
        modulus,
        BN254_NUM_LIMBS,
        BN254_LIMB_BITS,
        tester.memory_controller().borrow().range_checker.bus(),
    );

    let x = bn254_fq12_to_biguint_vec(&bn254_fq12_random(59));
    let y = bn254_fq12_to_biguint_vec(&bn254_fq12_random(3));

    test_fp12_fn::<BN254_NUM_LIMBS, BN254_LIMB_BITS>(
        tester,
        expr,
        Bn254Fp12Opcode::default_offset(),
        Fp12Opcode::SUB as usize,
        "Bn254Fp12Sub",
        x,
        y,
        12,
    );
}

#[test]
fn test_fp12_mul_bn254() {
    let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = BN254.MODULUS.clone();
    let xi = BN254.XI;
    let expr = fp12_mul_expr(
        modulus,
        BN254_NUM_LIMBS,
        BN254_LIMB_BITS,
        tester.memory_controller().borrow().range_checker.bus(),
        xi,
    );

    let x = bn254_fq12_to_biguint_vec(&bn254_fq12_random(5));
    let y = bn254_fq12_to_biguint_vec(&bn254_fq12_random(25));

    test_fp12_fn::<BN254_NUM_LIMBS, BN254_LIMB_BITS>(
        tester,
        expr,
        Bn254Fp12Opcode::default_offset(),
        Fp12Opcode::MUL as usize,
        "Bn254Fp12Mul",
        x,
        y,
        33,
    );
}

#[test]
fn test_fp12_add_bls12381() {
    let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = BLS12381.MODULUS.clone();
    let expr = fp12_add_expr(
        modulus,
        BLS12381_NUM_LIMBS,
        BLS12381_LIMB_BITS,
        tester.memory_controller().borrow().range_checker.bus(),
    );

    let x = bls12381_fq12_random(3);
    let y = bls12381_fq12_random(99);

    test_fp12_fn::<BLS12381_NUM_LIMBS, BLS12381_LIMB_BITS>(
        tester,
        expr,
        Bls12381Fp12Opcode::default_offset(),
        Fp12Opcode::ADD as usize,
        "Bls12381Fp12Add",
        x,
        y,
        12,
    );
}

#[test]
fn test_fp12_sub_bls12381() {
    let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = BLS12381.MODULUS.clone();
    let expr = fp12_sub_expr(
        modulus,
        BLS12381_NUM_LIMBS,
        BLS12381_LIMB_BITS,
        tester.memory_controller().borrow().range_checker.bus(),
    );

    let x = bls12381_fq12_random(8);
    let y = bls12381_fq12_random(9);

    test_fp12_fn::<BLS12381_NUM_LIMBS, BLS12381_LIMB_BITS>(
        tester,
        expr,
        Bls12381Fp12Opcode::default_offset(),
        Fp12Opcode::SUB as usize,
        "Bls12381Fp12Sub",
        x,
        y,
        12,
    );
}

// NOTE[yj]: This test requires RUST_MIN_STACK=8388608 to run without overflowing the stack, so it is ignored by the test runner for now
#[test]
#[ignore]
fn test_fp12_mul_bls12381() {
    let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = BLS12381.MODULUS.clone();
    let xi = BLS12381.XI;
    let expr = fp12_mul_expr(
        modulus,
        BLS12381_NUM_LIMBS,
        BLS12381_LIMB_BITS,
        tester.memory_controller().borrow().range_checker.bus(),
        xi,
    );

    let x = bls12381_fq12_random(5);
    let y = bls12381_fq12_random(25);

    test_fp12_fn::<BLS12381_NUM_LIMBS, BLS12381_LIMB_BITS>(
        tester,
        expr,
        Bls12381Fp12Opcode::default_offset(),
        Fp12Opcode::MUL as usize,
        "Bls12381Fp12Mul",
        x,
        y,
        82,
    );
}

// #[test]
// fn test_mul_013_by_013() {
//     const NUM_LIMBS: usize = 32;
//     const LIMB_BITS: usize = 8;

//     let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
//     let modulus = BN254.MODULUS.clone();
//     let xi = BN254.XI;
//     let expr = mul_013_by_013_expr(
//         modulus,
//         NUM_LIMBS,
//         LIMB_BITS,
//         tester.memory_controller().borrow().range_checker.bus(),
//         xi,
//     );

//     let x = bn254_fq12_to_biguint_vec(&bn254_fq12_random(1));
//     let y = bn254_fq12_to_biguint_vec(&bn254_fq12_random(2));

//     test_fp12_fn::<BN254_NUM_LIMBS, BN254_LIMB_BITS>(
//         tester,
//         expr,
//         Bn254Fp12Opcode::default_offset(),
//         Fp12Opcode::MUL_013_BY_013 as usize,
//         "Bn254Fp12Mul013By013",
//         x,
//         y,
//         33,
//     );
// }

#[test]
fn test_mul_013_by_013() {
    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;

    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let expr = mul_013_by_013_expr(
        BN254.MODULUS.clone(),
        NUM_LIMBS,
        LIMB_BITS,
        tester.memory_controller().borrow().range_checker.bus(),
        BN254.XI,
    );
    let core = FieldExpressionCoreChip::new(
        expr,
        Bn254Fp12Opcode::default_offset(),
        vec![Fp12Opcode::MUL_013_BY_013 as usize],
        tester.memory_controller().borrow().range_checker.clone(),
        "Bn254Fp12Mul013By013",
    );
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 4, 10, NUM_LIMBS, NUM_LIMBS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );

    let mut rng0 = StdRng::seed_from_u64(8);
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
    let line0 = point_to_013::<Fq, Fq2>(ec_pt_0);
    let line1 = point_to_013::<Fq, Fq2>(ec_pt_1);
    let input_line0 = vec![
        bn254_fq2_to_biguint_vec(&line0.b),
        bn254_fq2_to_biguint_vec(&line0.c),
    ]
    .concat();
    let input_line1 = vec![
        bn254_fq2_to_biguint_vec(&line1.b),
        bn254_fq2_to_biguint_vec(&line1.c),
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

    let r_cmp = ax_ecc_execution::curves::bn254::mul_013_by_013::<Fq, Fq2>(
        line0,
        line1,
        Fq2::new(Fq::from_raw([9, 0, 0, 0]), Fq::one()),
    );
    let r_cmp_bigint = r_cmp
        .map(|x| [bn254_fq_to_biguint(&x.c0), bn254_fq_to_biguint(&x.c1)])
        .concat();

    for i in 0..10 {
        assert_eq!(output[i], r_cmp_bigint[i]);
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
        chip.core.air.offset + Fp12Opcode::MUL_013_BY_013 as usize,
    );

    tester.execute(&mut chip, instruction);
    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}
