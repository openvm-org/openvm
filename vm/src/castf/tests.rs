use std::{
    array::{self},
    borrow::BorrowMut,
    sync::Arc,
};

use afs_primitives::var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip};
use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::{config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::{rngs::StdRng, Rng};

use super::{
    air::{FINAL_LIMB_SIZE, LIMB_SIZE},
    columns::CastFCols,
    CastFChip,
};
use crate::{
    arch::{chips::MachineChip, instructions::Opcode, testing::MachineChipTestBuilder},
    cpu::{trace::Instruction, RANGE_CHECKER_BUS},
};

type F = BabyBear;

fn generate_uint_number(rng: &mut StdRng) -> [u32; 4] {
    array::from_fn(|i| match i {
        0 | 1 | 2 => rng.gen_range(0..1 << LIMB_SIZE),
        3 => rng.gen_range(0..1 << FINAL_LIMB_SIZE),
        _ => unreachable!(),
    })
}

fn prepare_castf_rand_write_execute(
    tester: &mut MachineChipTestBuilder<F>,
    chip: &mut CastFChip<F>,
    x: [u32; 4],
    rng: &mut StdRng,
) {
    let address_space_range = || 1usize..=2;
    let address_range = || 0usize..1 << 29;

    let operand1 = x;

    let as_x = rng.gen_range(address_space_range()); // d
    let as_y = rng.gen_range(address_space_range()); // e
    let address_x = rng.gen_range(address_range()); // op_a
    let address_y = rng.gen_range(address_range()); // op_b

    let operand1_f = x.clone().map(F::from_canonical_u32);

    tester.write::<4>(as_x, address_x, operand1_f);
    let y = CastFChip::<F>::solve(&operand1);

    tester.execute(
        chip,
        Instruction::from_usize(Opcode::CASTF, [address_x, address_y, 0, as_x, as_y]),
    );
    assert_eq!([F::from_canonical_u32(y)], tester.read(as_y, address_y));
}

#[test]
fn castf_rand_test() {
    let mut rng = create_seeded_rng();
    let mut tester = MachineChipTestBuilder::default();
    let bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, LIMB_SIZE);
    let range_checker_chip = Arc::new(VariableRangeCheckerChip::new(bus));
    let mut chip = CastFChip::<F>::new(
        tester.execution_bus(),
        tester.memory_chip(),
        range_checker_chip.clone(),
    );
    let num_tests: usize = 10;

    for _ in 0..num_tests {
        let x = generate_uint_number(&mut rng);
        prepare_castf_rand_write_execute(&mut tester, &mut chip, x, &mut rng);
    }

    let tester = tester
        .build()
        .load(chip)
        .load(range_checker_chip)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn negative_castf_overflow_test() {
    let bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, LIMB_SIZE);
    let range_checker_chip = Arc::new(VariableRangeCheckerChip::new(bus));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = CastFChip::<F>::new(
        tester.execution_bus(),
        tester.memory_chip(),
        range_checker_chip.clone(),
    );

    let mut rng = create_seeded_rng();
    let x = generate_uint_number(&mut rng);
    prepare_castf_rand_write_execute(&mut tester, &mut chip, x, &mut rng);

    let air = chip.air.clone();
    let mut trace = chip.generate_trace();
    let cols: &mut CastFCols<F> = trace.values[..].borrow_mut();
    cols.io.x[3] = F::from_canonical_u32(rng.gen_range(1 << FINAL_LIMB_SIZE..1 << LIMB_SIZE));

    let range_air = range_checker_chip.air.clone();
    let range_trace = range_checker_chip.generate_trace();

    disable_debug_builder();
    assert_eq!(
        run_simple_test_no_pis(vec![&air, &range_air], vec![trace, range_trace],),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it didn't"
    );
}

#[test]
fn negative_castf_memread_test() {
    let bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, LIMB_SIZE);
    let range_checker_chip = Arc::new(VariableRangeCheckerChip::new(bus));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = CastFChip::<F>::new(
        tester.execution_bus(),
        tester.memory_chip(),
        range_checker_chip.clone(),
    );

    let mut rng = create_seeded_rng();
    let x = generate_uint_number(&mut rng);
    prepare_castf_rand_write_execute(&mut tester, &mut chip, x, &mut rng);

    let air = chip.air.clone();
    let mut trace = chip.generate_trace();
    let cols: &mut CastFCols<F> = trace.values[..].borrow_mut();
    cols.io.op_a = cols.io.op_a + F::one();

    let range_air = range_checker_chip.air.clone();
    let range_trace = range_checker_chip.generate_trace();

    disable_debug_builder();
    assert_eq!(
        run_simple_test_no_pis(vec![&air, &range_air], vec![trace, range_trace],),
        Err(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it didn't"
    );
}

#[test]
fn negative_castf_memwrite_test() {
    let bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, LIMB_SIZE);
    let range_checker_chip = Arc::new(VariableRangeCheckerChip::new(bus));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = CastFChip::<F>::new(
        tester.execution_bus(),
        tester.memory_chip(),
        range_checker_chip.clone(),
    );

    let mut rng = create_seeded_rng();
    let x = generate_uint_number(&mut rng);
    prepare_castf_rand_write_execute(&mut tester, &mut chip, x, &mut rng);

    let air = chip.air.clone();
    let mut trace = chip.generate_trace();
    let cols: &mut CastFCols<F> = trace.values[..].borrow_mut();
    cols.io.op_b = cols.io.op_b + F::one();

    let range_air = range_checker_chip.air.clone();
    let range_trace = range_checker_chip.generate_trace();

    disable_debug_builder();
    assert_eq!(
        run_simple_test_no_pis(vec![&air, &range_air], vec![trace, range_trace],),
        Err(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it didn't"
    );
}

#[test]
fn negative_castf_as_test() {
    let bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, LIMB_SIZE);
    let range_checker_chip = Arc::new(VariableRangeCheckerChip::new(bus));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = CastFChip::<F>::new(
        tester.execution_bus(),
        tester.memory_chip(),
        range_checker_chip.clone(),
    );

    let mut rng = create_seeded_rng();
    let x = generate_uint_number(&mut rng);
    prepare_castf_rand_write_execute(&mut tester, &mut chip, x, &mut rng);

    let air = chip.air.clone();
    let mut trace = chip.generate_trace();
    let cols: &mut CastFCols<F> = trace.values[..].borrow_mut();
    cols.io.d = cols.io.d + F::one();

    let range_air = range_checker_chip.air.clone();
    let range_trace = range_checker_chip.generate_trace();

    disable_debug_builder();
    assert_eq!(
        run_simple_test_no_pis(vec![&air, &range_air], vec![trace, range_trace],),
        Err(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it didn't"
    );
}
