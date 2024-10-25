use std::str::FromStr;

use afs_primitives::bigint::utils::secp256k1_coord_prime;
use axvm_instructions::UsizeOpcode;
use num_bigint_dig::BigUint;
use num_traits::FromPrimitive;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::{ec_add_ne_expr, ec_double_expr};
use crate::{
    arch::{instructions::EccOpcode, testing::VmChipTestBuilder, VmChipWrapper},
    intrinsics::field_expression::FieldExpressionCoreChip,
    rv32im::adapters::{Rv32VecHeapAdapterChip, RV32_REGISTER_NUM_LIMBS},
    system::program::Instruction,
    utils::biguint_to_limbs,
};

const NUM_LIMBS: usize = 32;
const LIMB_BITS: usize = 8;
type F = BabyBear;

#[test]
fn test_add_ne() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = secp256k1_coord_prime();
    let expr = ec_add_ne_expr(
        modulus,
        NUM_LIMBS,
        LIMB_BITS,
        tester.memory_controller().borrow().range_checker.bus(),
    );
    let core = FieldExpressionCoreChip::new(
        expr,
        EccOpcode::default_offset(),
        vec![EccOpcode::EC_ADD_NE as usize],
        tester.memory_controller().borrow().range_checker.clone(),
        "EcAddNe",
    );
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 2, 2, NUM_LIMBS, NUM_LIMBS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );

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

    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

    let r = chip
        .core
        .air
        .expr
        .execute(vec![p1_x, p1_y, p2_x, p2_y], vec![]);
    assert_eq!(r.len(), 3); // lambda, x3, y3
    assert_eq!(r[1], SampleEcPoints[2].0);
    assert_eq!(r[2], SampleEcPoints[2].1);

    let ptr_as = 1;
    let addr_ptr1 = 0;
    let addr_ptr2 = 3 * RV32_REGISTER_NUM_LIMBS;
    let addr_ptr3 = 6 * RV32_REGISTER_NUM_LIMBS;

    let data_as = 2;
    let address1 = 0u32;
    let address2 = 128u32;
    let address3 = 256u32;
    let mut write_reg = |reg_addr, value: u32| {
        tester.write(
            ptr_as,
            reg_addr,
            value.to_le_bytes().map(BabyBear::from_canonical_u8),
        );
    };

    write_reg(addr_ptr1, address1);
    write_reg(addr_ptr2, address2);
    write_reg(addr_ptr3, address3);
    tester.write(data_as, address1 as usize, p1_x_limbs);
    tester.write(data_as, address1 as usize + NUM_LIMBS, p1_y_limbs);
    tester.write(data_as, address2 as usize, p2_x_limbs);
    tester.write(data_as, address2 as usize + NUM_LIMBS, p2_y_limbs);

    let instruction = Instruction::from_isize(
        chip.core.air.offset + EccOpcode::EC_ADD_NE as usize,
        addr_ptr3 as isize,
        addr_ptr1 as isize,
        addr_ptr2 as isize,
        ptr_as as isize,
        data_as as isize,
    );
    tester.execute(&mut chip, instruction);

    let tester = tester.build().load(chip).finalize();

    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_double() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = secp256k1_coord_prime();
    let expr = ec_double_expr(
        modulus,
        NUM_LIMBS,
        LIMB_BITS,
        tester.memory_controller().borrow().range_checker.bus(),
    );
    let core = FieldExpressionCoreChip::new(
        expr,
        EccOpcode::default_offset(),
        vec![EccOpcode::EC_DOUBLE as usize],
        tester.memory_controller().borrow().range_checker.clone(),
        "EcDouble",
    );
    let adapter = Rv32VecHeapAdapterChip::<F, 1, 2, 2, NUM_LIMBS, NUM_LIMBS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );

    let (p1_x, p1_y) = SampleEcPoints[1].clone();
    let p1_x_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p1_y_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);

    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

    let r = chip.core.air.expr.execute(vec![p1_x, p1_y], vec![]);
    assert_eq!(r.len(), 3); // lambda, x3, y3
    assert_eq!(r[1], SampleEcPoints[3].0);
    assert_eq!(r[2], SampleEcPoints[3].1);
    let ptr_as = 1;
    let addr_ptr1 = 0;
    let addr_ptr2 = 0; // unused
    let addr_ptr3 = 6 * RV32_REGISTER_NUM_LIMBS;

    let data_as = 2;
    let address1 = 0u32;
    let address3 = 256u32;
    let mut write_reg = |reg_addr, value: u32| {
        tester.write(
            ptr_as,
            reg_addr,
            value.to_le_bytes().map(BabyBear::from_canonical_u8),
        );
    };
    write_reg(addr_ptr1, address1);
    write_reg(addr_ptr3, address3);
    tester.write(data_as, address1 as usize, p1_x_limbs);
    tester.write(data_as, address1 as usize + NUM_LIMBS, p1_y_limbs);

    let instruction = Instruction::from_isize(
        chip.core.air.offset + EccOpcode::EC_DOUBLE as usize,
        addr_ptr3 as isize,
        addr_ptr1 as isize,
        addr_ptr2 as isize,
        ptr_as as isize,
        data_as as isize,
    );
    tester.execute(&mut chip, instruction);
    let tester = tester.build().load(chip).finalize();

    tester.simple_test().expect("Verification failed");
}

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
        let x3 = BigUint::from_str("109562500687829935604265064386702914290271628241900466384583316550888437213118").unwrap();
        let y3 = BigUint::from_str(
            "54782835737747434227939451500021052510566980337100013600092875738315717035444",
        )
        .unwrap();

        // This is the double of (x2, y2).
        let x4 = BigUint::from_str(
            "23158417847463239084714197001737581570653996933128112807891516801581766934331").unwrap();
        let y4 = BigUint::from_str(
            "25821202496262252602076867233819373685524812798827903993634621255495124276396",
        )
        .unwrap();

        // This is the sum of (x3, y3) and (x4, y4).
        let x5 = BigUint::from_str("88733411122275068320336854419305339160905807011607464784153110222112026831518").unwrap();
        let y5 = BigUint::from_str(
            "69295025707265750480609159026651746584753914962418372690287755773539799515030",
        )
        .unwrap();

        vec![(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]
    };
}
