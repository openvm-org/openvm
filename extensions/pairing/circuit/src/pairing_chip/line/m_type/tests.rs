use halo2curves_axiom::{
    bls12_381::{Fq, Fq12, Fq2, G1Affine},
    ff::Field,
};
use openvm_circuit::arch::testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_ecc_guest::AffinePoint;
use openvm_instructions::{riscv::RV32_CELL_BITS, LocalOpcode};
use openvm_mod_circuit_builder::{test_utils::*, ExprBuilderConfig};
use openvm_pairing_guest::{
    bls12_381::{BLS12_381_LIMB_BITS, BLS12_381_MODULUS, BLS12_381_NUM_LIMBS, BLS12_381_XI_ISIZE},
    halo2curves_shims::bls12_381::{tangent_line_023, Bls12_381},
    pairing::LineMulMType,
};
use openvm_pairing_transpiler::PairingOpcode;
use openvm_rv32_adapters::{
    rv32_write_heap_default_with_increment, Rv32VecHeapAdapterChip, Rv32VecHeapTwoReadsAdapterChip,
};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rand::{rngs::StdRng, SeedableRng};

use super::*;

type F = BabyBear;
const NUM_LIMBS: usize = 48;
const LIMB_BITS: usize = 8;
const BLOCK_SIZE: usize = 16;

#[test]
fn test_mul_023_by_023() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 12, 30, BLOCK_SIZE, BLOCK_SIZE>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_bridge(),
        tester.address_bits(),
        bitwise_chip.clone(),
    );
    let mut chip = EcLineMul023By023Chip::new(
        adapter,
        tester.memory_controller().borrow().range_checker.clone(),
        ExprBuilderConfig {
            modulus: BLS12_381_MODULUS.clone(),
            num_limbs: BLS12_381_NUM_LIMBS,
            limb_bits: BLS12_381_LIMB_BITS,
        },
        BLS12_381_XI_ISIZE,
        PairingOpcode::CLASS_OFFSET,
        tester.offline_memory_mutex_arc(),
    );

    let mut rng0 = StdRng::seed_from_u64(15);
    let mut rng1 = StdRng::seed_from_u64(95);
    let rnd_pt_0 = G1Affine::random(&mut rng0);
    let rnd_pt_1 = G1Affine::random(&mut rng1);
    let ec_pt_0 = AffinePoint::<Fq> {
        x: rnd_pt_0.x,
        y: rnd_pt_0.y,
    };
    let ec_pt_1 = AffinePoint::<Fq> {
        x: rnd_pt_1.x,
        y: rnd_pt_1.y,
    };
    let line0 = tangent_line_023::<Fq, Fq2>(ec_pt_0);
    let line1 = tangent_line_023::<Fq, Fq2>(ec_pt_1);
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

    let r_cmp = Bls12_381::mul_023_by_023(&line0, &line1);
    let r_cmp_bigint = r_cmp
        .map(|x| [bls12381_fq_to_biguint(x.c0), bls12381_fq_to_biguint(x.c1)])
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

    let instruction = rv32_write_heap_default_with_increment(
        &mut tester,
        input_line0_limbs,
        input_line1_limbs,
        512,
        chip.0.core.air.offset + PairingOpcode::MUL_023_BY_023 as usize,
    );

    tester.execute(&mut chip, &instruction);
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

// NOTE[yj]: this test requires `RUST_MIN_STACK=8388608` to run otherwise it will overflow the stack
#[test]
#[ignore]
fn test_mul_by_02345() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let adapter = Rv32VecHeapTwoReadsAdapterChip::<F, 36, 30, 36, BLOCK_SIZE, BLOCK_SIZE>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_bridge(),
        tester.address_bits(),
        bitwise_chip.clone(),
    );
    let mut chip = EcLineMulBy02345Chip::new(
        adapter,
        tester.memory_controller().borrow().range_checker.clone(),
        ExprBuilderConfig {
            modulus: BLS12_381_MODULUS.clone(),
            num_limbs: BLS12_381_NUM_LIMBS,
            limb_bits: BLS12_381_LIMB_BITS,
        },
        BLS12_381_XI_ISIZE,
        PairingOpcode::CLASS_OFFSET,
        tester.offline_memory_mutex_arc(),
    );

    let mut rng = StdRng::seed_from_u64(19);
    let f = Fq12::random(&mut rng);
    let x0 = Fq2::random(&mut rng);
    let x2 = Fq2::random(&mut rng);
    let x3 = Fq2::random(&mut rng);
    let x4 = Fq2::random(&mut rng);
    let x5 = Fq2::random(&mut rng);

    let input_f = bls12381_fq12_to_biguint_vec(f);
    let input_x = [
        bls12381_fq2_to_biguint_vec(x0),
        bls12381_fq2_to_biguint_vec(x2),
        bls12381_fq2_to_biguint_vec(x3),
        bls12381_fq2_to_biguint_vec(x4),
        bls12381_fq2_to_biguint_vec(x5),
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

    let r_cmp = Bls12_381::mul_by_02345(&f, &[x0, x2, x3, x4, x5]);
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

    let instruction = rv32_write_heap_default_with_increment(
        &mut tester,
        input_f_limbs,
        input_x_limbs,
        1024,
        chip.0.core.air.offset + PairingOpcode::MUL_BY_02345 as usize,
    );

    tester.execute(&mut chip, &instruction);
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}
