use std::sync::Arc;

use ax_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, BitwiseOperationLookupChip,
};
use ax_ecc_execution::{
    common::{EcPoint, UnevaluatedLine},
    curves::bn254::tangent_line_013,
};
use ax_ecc_primitives::{
    field_expression::ExprBuilderConfig,
    test_utils::{bn254_fq12_to_biguint_vec, bn254_fq2_to_biguint_vec, bn254_fq_to_biguint},
};
use axvm_ecc_constants::BN254;
use axvm_instructions::{riscv::RV32_CELL_BITS, PairingOpcode, UsizeOpcode};
use halo2curves_axiom::{
    bn256::{Fq, Fq12, Fq2, G1Affine},
    ff::Field,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    arch::{testing::VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    intrinsics::ecc::pairing::{EcLineMul013By013Chip, EcLineMulBy01234Chip, EvaluateLineChip},
    rv32im::adapters::{Rv32VecHeapAdapterChip, Rv32VecHeapTwoReadsAdapterChip},
    utils::{biguint_to_limbs, rv32_write_heap_default, rv32_write_heap_default_with_increment},
};

type F = BabyBear;
const NUM_LIMBS: usize = 32;
const LIMB_BITS: usize = 8;
const BLOCK_SIZE: usize = 32;

#[test]
fn test_mul_013_by_013() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 4, 10, BLOCK_SIZE, BLOCK_SIZE>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        bitwise_chip.clone(),
    );
    let mut chip = EcLineMul013By013Chip::new(
        adapter,
        tester.memory_controller(),
        ExprBuilderConfig {
            modulus: BN254.MODULUS.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        },
        BN254.XI,
        PairingOpcode::default_offset(),
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
    let line0 = tangent_line_013::<Fq, Fq2>(ec_pt_0);
    let line1 = tangent_line_013::<Fq, Fq2>(ec_pt_1);
    let input_line0 = [
        bn254_fq2_to_biguint_vec(line0.b),
        bn254_fq2_to_biguint_vec(line0.c),
    ]
    .concat();
    let input_line1 = [
        bn254_fq2_to_biguint_vec(line1.b),
        bn254_fq2_to_biguint_vec(line1.c),
    ]
    .concat();

    let vars = chip
        .0
        .core
        .expr()
        .execute([input_line0.clone(), input_line1.clone()].concat(), vec![]);
    let output_indices = chip.0.core.expr().builder.output_indices.clone();
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
        .map(|x| [bn254_fq_to_biguint(x.c0), bn254_fq_to_biguint(x.c1)])
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
        chip.0.core.air.offset + PairingOpcode::MUL_013_BY_013 as usize,
    );

    tester.execute(&mut chip, instruction);
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_mul_by_01234() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 12, 12, BLOCK_SIZE, BLOCK_SIZE>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        bitwise_chip.clone(),
    );
    let mut chip = EcLineMulBy01234Chip::new(
        adapter,
        tester.memory_controller(),
        ExprBuilderConfig {
            modulus: BN254.MODULUS.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        },
        BN254.XI,
        PairingOpcode::default_offset(),
    );

    let mut rng = StdRng::seed_from_u64(8);
    let f = Fq12::random(&mut rng);
    let mut rng = StdRng::seed_from_u64(12);
    let x0 = Fq2::random(&mut rng);
    let mut rng = StdRng::seed_from_u64(1);
    let x1 = Fq2::random(&mut rng);
    let mut rng = StdRng::seed_from_u64(5);
    let x2 = Fq2::random(&mut rng);
    let mut rng = StdRng::seed_from_u64(77);
    let x3 = Fq2::random(&mut rng);
    let mut rng = StdRng::seed_from_u64(31);
    let x4 = Fq2::random(&mut rng);

    let input_f = bn254_fq12_to_biguint_vec(f);
    let input_x = [
        bn254_fq2_to_biguint_vec(x0),
        bn254_fq2_to_biguint_vec(x1),
        bn254_fq2_to_biguint_vec(x2),
        bn254_fq2_to_biguint_vec(x3),
        bn254_fq2_to_biguint_vec(x4),
        bn254_fq2_to_biguint_vec(Fq2::zero()),
    ]
    .concat();

    let vars = chip
        .0
        .core
        .expr()
        .execute([input_f.clone(), input_x.clone()].concat(), vec![]);
    let output_indices = chip.0.core.expr().builder.output_indices.clone();
    let output = output_indices
        .iter()
        .map(|i| vars[*i].clone())
        .collect::<Vec<_>>();
    assert_eq!(output.len(), 12);

    let r_cmp =
        ax_ecc_execution::curves::bn254::mul_by_01234::<Fq, Fq2, Fq12>(f, [x0, x1, x2, x3, x4]);
    let r_cmp_bigint = bn254_fq12_to_biguint_vec(r_cmp);

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

    let instruction = rv32_write_heap_default_with_increment(
        &mut tester,
        input_f_limbs,
        input_x_limbs,
        512,
        chip.0.core.air.offset + PairingOpcode::MUL_BY_01234 as usize,
    );

    tester.execute(&mut chip, instruction);
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_evaluate_line() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: BN254.MODULUS.clone(),
        limb_bits: LIMB_BITS,
        num_limbs: NUM_LIMBS,
    };
    let adapter = Rv32VecHeapTwoReadsAdapterChip::<F, 4, 2, 4, BLOCK_SIZE, BLOCK_SIZE>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );
    let mut chip = EvaluateLineChip::new(
        adapter,
        tester.memory_controller(),
        config,
        PairingOpcode::default_offset(),
    );

    let mut rng = StdRng::seed_from_u64(42);
    let uneval_b = Fq2::random(&mut rng);
    let uneval_c = Fq2::random(&mut rng);
    let x_over_y = Fq::random(&mut rng);
    let y_inv = Fq::random(&mut rng);
    let mut inputs = vec![];
    inputs.extend(bn254_fq2_to_biguint_vec(uneval_b));
    inputs.extend(bn254_fq2_to_biguint_vec(uneval_c));
    inputs.push(bn254_fq_to_biguint(x_over_y));
    inputs.push(bn254_fq_to_biguint(y_inv));
    let input_limbs = inputs
        .iter()
        .map(|x| {
            biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32)
        })
        .collect();

    let uneval: UnevaluatedLine<Fq, Fq2> = UnevaluatedLine {
        b: uneval_b,
        c: uneval_c,
    };
    let evaluated = uneval.evaluate(x_over_y, y_inv);

    let result = chip.0.core.expr().execute_with_output(inputs, vec![]);
    assert_eq!(result.len(), 4);
    assert_eq!(result[0], bn254_fq_to_biguint(evaluated.b.c0));
    assert_eq!(result[1], bn254_fq_to_biguint(evaluated.b.c1));
    assert_eq!(result[2], bn254_fq_to_biguint(evaluated.c.c0));
    assert_eq!(result[3], bn254_fq_to_biguint(evaluated.c.c1));

    let instruction = rv32_write_heap_default(
        &mut tester,
        input_limbs,
        vec![],
        chip.0.core.air.offset + PairingOpcode::EVALUATE_LINE as usize,
    );

    tester.execute(&mut chip, instruction);
    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}
