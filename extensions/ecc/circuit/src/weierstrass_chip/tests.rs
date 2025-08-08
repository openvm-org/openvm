use std::{str::FromStr, sync::Arc};

use num_bigint::BigUint;
use num_traits::{FromPrimitive, Num, Zero};
use openvm_circuit::arch::{
    testing::{TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    MatrixRecordArena,
};
use openvm_circuit_primitives::{
    bigint::utils::{secp256k1_coord_prime, secp256r1_coord_prime},
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
};
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::{riscv::RV32_CELL_BITS, LocalOpcode};
use openvm_mod_circuit_builder::{test_utils::biguint_to_limbs, ExprBuilderConfig, FieldExpr};
use openvm_rv32_adapters::rv32_write_heap_default;
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::{
    get_ec_addne_air, get_ec_addne_chip, get_ec_addne_step, get_ec_double_air, get_ec_double_chip,
    get_ec_double_step, EcDoubleExecutor, WeierstrassAir, WeierstrassChip,
};

const NUM_LIMBS: usize = 32;
const LIMB_BITS: usize = 8;
const BLOCK_SIZE: usize = 32;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

lazy_static::lazy_static! {
    // Sample points got from https://asecuritysite.com/ecc/ecc_points2 and
    // https://learnmeabitcoin.com/technical/cryptography/elliptic-curve/#add
    pub static ref SampleEcPoints: Vec<(BigUint, BigUint)> = {
        let x1 = BigUint::from_u32(1).unwrap();
        let y1 = BigUint::from_str(
            "29896722852569046015560700294576055776214335159245303116488692907525646231534",
        )
        .unwrap();
        let x2 = BigUint::from_u32(2).unwrap();
        let y2 = BigUint::from_str(
            "69211104694897500952317515077652022726490027694212560352756646854116994689233",
        )
        .unwrap();

        // This is the sum of (x1, y1) and (x2, y2).
        let x3 = BigUint::from_str(
            "109562500687829935604265064386702914290271628241900466384583316550888437213118",
        )
        .unwrap();
        let y3 = BigUint::from_str(
            "54782835737747434227939451500021052510566980337100013600092875738315717035444",
        )
        .unwrap();

        // This is the double of (x2, y2).
        let x4 = BigUint::from_str(
            "23158417847463239084714197001737581570653996933128112807891516801581766934331",
        )
        .unwrap();
        let y4 = BigUint::from_str(
            "25821202496262252602076867233819373685524812798827903993634621255495124276396",
        )
        .unwrap();

        // This is the sum of (x3, y3) and (x4, y4).
        let x5 = BigUint::from_str(
            "88733411122275068320336854419305339160905807011607464784153110222112026831518",
        )
        .unwrap();
        let y5 = BigUint::from_str(
            "69295025707265750480609159026651746584753914962418372690287755773539799515030",
        )
        .unwrap();

        vec![(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]
    };
}

fn prime_limbs(expr: &FieldExpr) -> Vec<BabyBear> {
    expr.prime_limbs
        .iter()
        .map(|n| BabyBear::from_canonical_usize(*n))
        .collect::<Vec<_>>()
}

type WeierstrassHarness = TestChipHarness<
    F,
    EcDoubleExecutor<2, BLOCK_SIZE>,
    WeierstrassAir<1, 2, BLOCK_SIZE>,
    WeierstrassChip<F, 1, 2, BLOCK_SIZE>,
    MatrixRecordArena<F>,
>;

fn create_test_double_chips(
    tester: &VmChipTestBuilder<F>,
    config: ExprBuilderConfig,
    offset: usize,
    a_biguint: BigUint,
) -> (
    WeierstrassHarness,
    (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let air = get_ec_double_air(
        tester.execution_bridge(),
        tester.memory_bridge(),
        config.clone(),
        tester.range_checker().bus(),
        bitwise_bus,
        tester.address_bits(),
        offset,
        a_biguint.clone(),
    );
    let executor = get_ec_double_step(
        config.clone(),
        tester.range_checker().bus(),
        tester.address_bits(),
        offset,
        a_biguint.clone(),
    );
    let chip = get_ec_double_chip(
        config.clone(),
        tester.memory_helper(),
        tester.range_checker(),
        bitwise_chip.clone(),
        tester.address_bits(),
        a_biguint,
    );
    let harness = WeierstrassHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (bitwise_chip.air, bitwise_chip))
}

#[test]
fn test_add_ne() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: secp256k1_coord_prime(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let air = get_ec_addne_air::<2, BLOCK_SIZE>(
        tester.execution_bridge(),
        tester.memory_bridge(),
        config.clone(),
        tester.range_checker().bus(),
        bitwise_bus,
        tester.address_bits(),
        Rv32WeierstrassOpcode::CLASS_OFFSET,
    );
    let executor = get_ec_addne_step::<2, BLOCK_SIZE>(
        config.clone(),
        tester.range_checker().bus(),
        tester.address_bits(),
        Rv32WeierstrassOpcode::CLASS_OFFSET,
    );
    let chip = get_ec_addne_chip::<F, 2, BLOCK_SIZE>(
        config.clone(),
        tester.memory_helper(),
        tester.range_checker(),
        bitwise_chip.clone(),
        tester.address_bits(),
    );

    let mut harness = TestChipHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    assert_eq!(harness.executor.expr.builder.num_variables, 3); // lambda, x3, y3

    let (p1_x, p1_y) = SampleEcPoints[0].clone();
    let (p2_x, p2_y) = SampleEcPoints[1].clone();

    let p1_x_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p1_y_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p2_x_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p2_x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p2_y_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p2_y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);

    let r = harness
        .executor
        .expr
        .execute(vec![p1_x, p1_y, p2_x, p2_y], vec![true]);
    assert_eq!(r.len(), 3); // lambda, x3, y3
    assert_eq!(r[1], SampleEcPoints[2].0);
    assert_eq!(r[2], SampleEcPoints[2].1);

    let prime_limbs: [BabyBear; NUM_LIMBS] =
        prime_limbs(&harness.executor.expr).try_into().unwrap();
    let mut one_limbs = [BabyBear::ONE; NUM_LIMBS];
    one_limbs[0] = BabyBear::ONE;
    let setup_instruction = rv32_write_heap_default(
        &mut tester,
        vec![prime_limbs, one_limbs], // inputs[0] = prime, others doesn't matter
        vec![one_limbs, one_limbs],
        harness.executor.offset + Rv32WeierstrassOpcode::SETUP_EC_ADD_NE as usize,
    );
    tester.execute(&mut harness, &setup_instruction);

    let instruction = rv32_write_heap_default(
        &mut tester,
        vec![p1_x_limbs, p1_y_limbs],
        vec![p2_x_limbs, p2_y_limbs],
        harness.executor.offset + Rv32WeierstrassOpcode::EC_ADD_NE as usize,
    );

    tester.execute(&mut harness, &instruction);

    let tester = tester
        .build()
        .load(harness)
        .load_periphery((bitwise_chip.air, bitwise_chip))
        .finalize();

    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_double() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: secp256k1_coord_prime(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };

    let (mut harness, bitwise) = create_test_double_chips(
        &tester,
        config,
        Rv32WeierstrassOpcode::CLASS_OFFSET,
        BigUint::zero(),
    );

    let (p1_x, p1_y) = SampleEcPoints[1].clone();
    let p1_x_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p1_y_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);

    assert_eq!(harness.executor.expr.builder.num_variables, 3); // lambda, x3, y3

    let r = harness.executor.expr.execute(vec![p1_x, p1_y], vec![true]);
    assert_eq!(r.len(), 3); // lambda, x3, y3
    assert_eq!(r[1], SampleEcPoints[3].0);
    assert_eq!(r[2], SampleEcPoints[3].1);

    let prime_limbs: [BabyBear; NUM_LIMBS] =
        prime_limbs(&harness.executor.expr).try_into().unwrap();
    let a_limbs = [BabyBear::ZERO; NUM_LIMBS];
    let setup_instruction = rv32_write_heap_default(
        &mut tester,
        vec![prime_limbs, a_limbs], /* inputs[0] = prime, inputs[1] = a coeff of weierstrass
                                     * equation */
        vec![],
        harness.executor.offset + Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
    );
    tester.execute(&mut harness, &setup_instruction);

    let instruction = rv32_write_heap_default(
        &mut tester,
        vec![p1_x_limbs, p1_y_limbs],
        vec![],
        harness.executor.offset + Rv32WeierstrassOpcode::EC_DOUBLE as usize,
    );

    tester.execute(&mut harness, &instruction);
    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();

    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_p256_double() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: secp256r1_coord_prime(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };
    let a = BigUint::from_str_radix(
        "ffffffff00000001000000000000000000000000fffffffffffffffffffffffc",
        16,
    )
    .unwrap();

    let (mut harness, bitwise) = create_test_double_chips(
        &tester,
        config,
        Rv32WeierstrassOpcode::CLASS_OFFSET,
        a.clone(),
    );

    // Testing data from: http://point-at-infinity.org/ecc/nisttv
    let p1_x = BigUint::from_str_radix(
        "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
        16,
    )
    .unwrap();
    let p1_y = BigUint::from_str_radix(
        "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
        16,
    )
    .unwrap();
    let p1_x_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p1_y_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);

    assert_eq!(harness.executor.expr.builder.num_variables, 3); // lambda, x3, y3

    let r = harness.executor.expr.execute(vec![p1_x, p1_y], vec![true]);
    assert_eq!(r.len(), 3); // lambda, x3, y3
    let expected_double_x = BigUint::from_str_radix(
        "7CF27B188D034F7E8A52380304B51AC3C08969E277F21B35A60B48FC47669978",
        16,
    )
    .unwrap();
    let expected_double_y = BigUint::from_str_radix(
        "07775510DB8ED040293D9AC69F7430DBBA7DADE63CE982299E04B79D227873D1",
        16,
    )
    .unwrap();
    assert_eq!(r[1], expected_double_x);
    assert_eq!(r[2], expected_double_y);

    let prime_limbs: [BabyBear; NUM_LIMBS] =
        prime_limbs(&harness.executor.expr).try_into().unwrap();
    let a_limbs =
        biguint_to_limbs::<NUM_LIMBS>(a.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let setup_instruction = rv32_write_heap_default(
        &mut tester,
        vec![prime_limbs, a_limbs], /* inputs[0] = prime, inputs[1] = a coeff of weierstrass
                                     * equation */
        vec![],
        harness.executor.offset + Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
    );
    tester.execute(&mut harness, &setup_instruction);

    let instruction = rv32_write_heap_default(
        &mut tester,
        vec![p1_x_limbs, p1_y_limbs],
        vec![],
        harness.executor.offset + Rv32WeierstrassOpcode::EC_DOUBLE as usize,
    );

    tester.execute(&mut harness, &instruction);
    // Adding another row to make sure there are dummy rows, and that the dummy row constraints are
    // satisfied
    tester.execute(&mut harness, &instruction);
    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();

    tester.simple_test().expect("Verification failed");
}
