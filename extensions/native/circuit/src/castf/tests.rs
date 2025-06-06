use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::{
    testing::{memory::gen_pointer, VmChipTestBuilder},
    VmAirWrapper,
};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_native_compiler::CastfOpcode;
use openvm_stark_backend::{
    p3_field::FieldAlgebra, utils::disable_debug_builder, verifier::VerificationError, Chip,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
    p3_baby_bear::BabyBear, utils::create_seeded_rng,
};
use rand::{rngs::StdRng, Rng};

use super::{
    super::adapters::convert_adapter::{ConvertAdapterAir, ConvertAdapterCols, ConvertAdapterStep},
    CastF, CastFChip, CastFCoreAir, CastFCoreCols, CastFStep, FINAL_LIMB_BITS, LIMB_BITS,
};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

fn create_test_chip(tester: &VmChipTestBuilder<F>) -> CastFChip<F> {
    let mut chip = CastFChip::<F>::new(
        VmAirWrapper::new(
            ConvertAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
            CastFCoreAir::new(tester.range_checker().bus()),
        ),
        CastFStep::new(
            ConvertAdapterStep::<1, 4>::new(),
            tester.range_checker().clone(),
        ),
        tester.memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);

    chip
}

fn generate_uint_number(rng: &mut StdRng) -> u32 {
    rng.gen_range(0..(1 << 30) - 1)
}

fn prepare_castf_rand_write_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut CastFChip<F>,
    y: u32,
    rng: &mut StdRng,
) {
    let operand1 = y;

    let as_x = 2usize; // d
    let as_y = 4usize; // e
    let address_x = gen_pointer(rng, 32); // a
    let address_y = gen_pointer(rng, 32); // b

    let operand1_f = F::from_canonical_u32(y);

    tester.memory.write(as_y, address_y, [operand1_f]);
    let x = CastF::solve(operand1);

    tester.execute(
        chip,
        &Instruction::from_usize(
            CastfOpcode::CASTF.global_opcode(),
            [address_x, address_y, 0, as_x, as_y],
        ),
    );
    assert_eq!(
        x.map(F::from_canonical_u8),
        tester.read::<4>(as_x, address_x)
    );
}

#[test]
fn castf_rand_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut chip = create_test_chip(&tester);
    let num_tests: usize = 3;

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
    let range_checker_chip = tester.range_checker();
    let mut chip = create_test_chip(&tester);

    let mut rng = create_seeded_rng();
    let y = generate_uint_number(&mut rng);
    prepare_castf_rand_write_execute(&mut tester, &mut chip, y, &mut rng);
    tester.build();

    let chip_air = chip.air();
    let mut chip_input = chip.generate_air_proof_input();
    let trace = chip_input.raw.common_main.as_mut().unwrap();
    let row = trace.row_mut(0);
    let cols: &mut CastFCoreCols<F> = row
        .split_at_mut(ConvertAdapterCols::<F, 1, 4>::width())
        .1
        .borrow_mut();
    cols.out_val[3] = F::from_canonical_u32(rng.gen_range(1 << FINAL_LIMB_BITS..1 << LIMB_BITS));

    let rc_air = range_checker_chip.air();
    let rc_p_input = range_checker_chip.generate_air_proof_input();

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_test_fast(
            vec![chip_air, rc_air],
            vec![chip_input, rc_p_input]
        )
        .err(),
        Some(VerificationError::ChallengePhaseError),
        "Expected verification to fail, but it didn't"
    );
}

#[test]
fn negative_castf_memread_test() {
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let mut chip = create_test_chip(&tester);

    let mut rng = create_seeded_rng();
    let y = generate_uint_number(&mut rng);
    prepare_castf_rand_write_execute(&mut tester, &mut chip, y, &mut rng);
    tester.build();

    let chip_air = chip.air();
    let mut chip_input = chip.generate_air_proof_input();
    let trace = chip_input.raw.common_main.as_mut().unwrap();
    let row = trace.row_mut(0);
    let cols: &mut ConvertAdapterCols<F, 1, 4> = row
        .split_at_mut(ConvertAdapterCols::<F, 1, 4>::width())
        .0
        .borrow_mut();
    cols.b_pointer += F::ONE;

    let rc_air = range_checker_chip.air();
    let rc_p_input = range_checker_chip.generate_air_proof_input();

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_test_fast(
            vec![chip_air, rc_air],
            vec![chip_input, rc_p_input]
        )
        .err(),
        Some(VerificationError::ChallengePhaseError),
        "Expected verification to fail, but it didn't"
    );
}

#[test]
fn negative_castf_memwrite_test() {
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let mut chip = create_test_chip(&tester);

    let mut rng = create_seeded_rng();
    let y = generate_uint_number(&mut rng);
    prepare_castf_rand_write_execute(&mut tester, &mut chip, y, &mut rng);
    tester.build();

    let chip_air = chip.air();
    let mut chip_input = chip.generate_air_proof_input();
    let trace = chip_input.raw.common_main.as_mut().unwrap();
    let row = trace.row_mut(0);
    let cols: &mut ConvertAdapterCols<F, 1, 4> = row
        .split_at_mut(ConvertAdapterCols::<F, 1, 4>::width())
        .0
        .borrow_mut();
    cols.a_pointer += F::ONE;

    let rc_air = range_checker_chip.air();
    let rc_p_input = range_checker_chip.generate_air_proof_input();

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_test_fast(
            vec![chip_air, rc_air],
            vec![chip_input, rc_p_input]
        )
        .err(),
        Some(VerificationError::ChallengePhaseError),
        "Expected verification to fail, but it didn't"
    );
}

#[test]
fn negative_castf_as_test() {
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let mut chip = create_test_chip(&tester);

    let mut rng = create_seeded_rng();
    let y = generate_uint_number(&mut rng);
    prepare_castf_rand_write_execute(&mut tester, &mut chip, y, &mut rng);
    tester.build();

    let chip_air = chip.air();
    let mut chip_input = chip.generate_air_proof_input();
    let trace = chip_input.raw.common_main.as_mut().unwrap();
    let row = trace.row_mut(0);
    let cols: &mut ConvertAdapterCols<F, 1, 4> = row
        .split_at_mut(ConvertAdapterCols::<F, 1, 4>::width())
        .0
        .borrow_mut();
    cols.a_pointer += F::ONE;

    let rc_air = range_checker_chip.air();
    let rc_p_input = range_checker_chip.generate_air_proof_input();

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_test_fast(
            vec![chip_air, rc_air],
            vec![chip_input, rc_p_input]
        )
        .err(),
        Some(VerificationError::ChallengePhaseError),
        "Expected verification to fail, but it didn't"
    );
}
