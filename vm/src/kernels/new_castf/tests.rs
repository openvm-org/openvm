use std::{borrow::BorrowMut, sync::Arc};

use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError, Chip};
use ax_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
    utils::create_seeded_rng,
};
use axvm_instructions::CastfOpcode;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::{rngs::StdRng, Rng};

use super::{NewCastFChip, NewCastFCoreChip};
use crate::{
    arch::{
        testing::{memory::gen_pointer, VmChipTestBuilder},
        VmChip,
    },
    kernels::{
        adapters::convert_adapter::{ConvertAdapterChip, ConvertAdapterCols},
        new_castf::{NewCastF, NewCastFCoreCols, FINAL_LIMB_SIZE, LIMB_SIZE},
    },
    system::program::Instruction,
};

type F = BabyBear;

fn generate_uint_number(rng: &mut StdRng) -> u32 {
    rng.gen_range(0..(1 << 30) - 1)
}

fn prepare_castf_rand_write_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut NewCastFChip<F>,
    y: u32,
    rng: &mut StdRng,
) {
    let address_space_range = || 1usize..=2;

    let operand1 = y;

    let as_x = rng.gen_range(address_space_range()); // d
    let as_y = rng.gen_range(address_space_range()); // e
    let address_x = gen_pointer(rng, 32); // op_a
    let address_y = gen_pointer(rng, 32); // op_b

    let operand1_f = F::from_canonical_u32(y);

    tester.write_cell(as_y, address_y, operand1_f);
    let x = NewCastF::solve(operand1);

    tester.execute(
        chip,
        Instruction::from_usize(
            CastfOpcode::CASTF as usize,
            [address_x, address_y, 0, as_x, as_y],
        ),
    );
    assert_eq!(
        x.map(F::from_canonical_u32),
        tester.read::<4>(as_x, address_x)
    );
}

#[test]
fn castf_rand_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let mut chip = NewCastFChip::<F>::new(
        ConvertAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        NewCastFCoreChip::new(range_checker_chip, 0),
        tester.memory_controller(),
    );
    let num_tests: usize = 1;

    for _ in 0..num_tests {
        let y = generate_uint_number(&mut rng);
        prepare_castf_rand_write_execute(&mut tester, &mut chip, y, &mut rng);
    }

    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn negative_castf_overflow_test() {
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let mut chip = NewCastFChip::<F>::new(
        ConvertAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        NewCastFCoreChip::new(range_checker_chip.clone(), 0),
        tester.memory_controller(),
    );

    let mut rng = create_seeded_rng();
    let y = generate_uint_number(&mut rng);
    prepare_castf_rand_write_execute(&mut tester, &mut chip, y, &mut rng);

    let air = chip.air();
    let range_air = range_checker_chip.air;
    let mut trace = chip.generate_trace();
    let row = trace.row_mut(0);
    let cols: &mut NewCastFCoreCols<F> = row
        .split_at_mut(ConvertAdapterCols::<F, 1, 4>::width())
        .1
        .borrow_mut();
    cols.out_val[3] = F::from_canonical_u32(rng.gen_range(1 << FINAL_LIMB_SIZE..1 << LIMB_SIZE));

    let range_trace = range_checker_chip.generate_trace();

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
            vec![air, Arc::new(range_air)],
            vec![trace, range_trace],
        )
        .err(),
        Some(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it didn't"
    );
}

#[test]
fn negative_castf_memread_test() {
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let mut chip = NewCastFChip::<F>::new(
        ConvertAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        NewCastFCoreChip::new(range_checker_chip.clone(), 0),
        tester.memory_controller(),
    );

    let mut rng = create_seeded_rng();
    let y = generate_uint_number(&mut rng);
    prepare_castf_rand_write_execute(&mut tester, &mut chip, y, &mut rng);

    let air = chip.air();
    let range_air = range_checker_chip.air;
    let mut trace = chip.generate_trace();
    let row = trace.row_mut(0);
    let cols: &mut ConvertAdapterCols<F, 1, 4> = row
        .split_at_mut(ConvertAdapterCols::<F, 1, 4>::width())
        .0
        .borrow_mut();
    cols.b_idx += F::one();

    let range_trace = range_checker_chip.generate_trace();

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
            vec![air, Arc::new(range_air)],
            vec![trace, range_trace],
        )
        .err(),
        Some(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it didn't"
    );
}

#[test]
fn negative_castf_memwrite_test() {
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let mut chip = NewCastFChip::<F>::new(
        ConvertAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        NewCastFCoreChip::new(range_checker_chip.clone(), 0),
        tester.memory_controller(),
    );

    let mut rng = create_seeded_rng();
    let y = generate_uint_number(&mut rng);
    prepare_castf_rand_write_execute(&mut tester, &mut chip, y, &mut rng);

    let air = chip.air();
    let range_air = range_checker_chip.air;
    let mut trace = chip.generate_trace();
    let row = trace.row_mut(0);
    let cols: &mut ConvertAdapterCols<F, 1, 4> = row
        .split_at_mut(ConvertAdapterCols::<F, 1, 4>::width())
        .0
        .borrow_mut();
    cols.a_idx += F::one();

    let range_trace = range_checker_chip.generate_trace();

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
            vec![air, Arc::new(range_air)],
            vec![trace, range_trace],
        )
        .err(),
        Some(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it didn't"
    );
}

#[test]
fn negative_castf_as_test() {
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let mut chip = NewCastFChip::<F>::new(
        ConvertAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        NewCastFCoreChip::new(range_checker_chip.clone(), 0),
        tester.memory_controller(),
    );

    let mut rng = create_seeded_rng();
    let y = generate_uint_number(&mut rng);
    prepare_castf_rand_write_execute(&mut tester, &mut chip, y, &mut rng);

    let air = chip.air();
    let range_air = range_checker_chip.air;
    let mut trace = chip.generate_trace();
    let row = trace.row_mut(0);
    let cols: &mut ConvertAdapterCols<F, 1, 4> = row
        .split_at_mut(ConvertAdapterCols::<F, 1, 4>::width())
        .0
        .borrow_mut();
    cols.a_as += F::one();

    let range_trace = range_checker_chip.generate_trace();

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
            vec![air, Arc::new(range_air)],
            vec![trace, range_trace],
        )
        .err(),
        Some(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it didn't"
    );
}
