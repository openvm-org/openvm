use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::{rngs::StdRng, Rng};

use super::UiChip;
use crate::{
    arch::{instructions::Opcode, testing::MachineChipTestBuilder},
    program::Instruction,
};

type F = BabyBear;

#[test]
fn solve_lui_sanity_test() {
    let b = 4097;
    let x = UiChip::<BabyBear>::solve_lui(b);
    assert_eq!(x, [0, 16, 0, 1]);
}

fn prepare_lui_write_execute(
    tester: &mut MachineChipTestBuilder<F>,
    chip: &mut UiChip<F>,
    a: usize,
    b: usize,
) {
    let x = UiChip::<F>::solve_lui(b as u32);

    tester.execute(chip, Instruction::from_usize(Opcode::LUI, [a, b]));
    assert_eq!(x.map(F::from_canonical_u32), tester.read::<4>(1usize, a));
}

fn generate_random_input(rng: &mut StdRng, address_bits: usize, imm_bits: usize) -> (usize, usize) {
    let address_range = || 0usize..1 << address_bits;
    let imm_range = || 0usize..1 << imm_bits;

    let a = rng.gen_range(address_range());
    let b = rng.gen_range(imm_range());

    (a, b)
}

#[test]
fn lui_test() {
    let mut rng = create_seeded_rng();

    let mut tester = MachineChipTestBuilder::default();
    let mut chip = UiChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
    );
    let num_tests: usize = 10;

    for _ in 0..num_tests {
        let (a, b) = generate_random_input(&mut rng, 29, 20);
        prepare_lui_write_execute(&mut tester, &mut chip, a, b);
    }

    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn negative_lui_invalid_imm_test() {
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = UiChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
    );

    // (1 << 20) + 1 exceeds the 20-bit bound
    prepare_lui_write_execute(&mut tester, &mut chip, 3, (1 << 20) + 1);

    let tester = tester.build().load(chip).finalize();
    disable_debug_builder();
    assert_eq!(
        tester.simple_test().err(),
        Some(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it didn't"
    );
}
