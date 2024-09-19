use std::borrow::BorrowMut;

use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::{
    any_rap_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::{rngs::StdRng, Rng};

use super::{columns::UiCols, UiChip};
use crate::{
    arch::{chips::MachineChip, instructions::Opcode, testing::MachineChipTestBuilder},
    cpu::trace::Instruction,
};

type F = BabyBear;

#[test]
fn solve_lui_sanity_test() {
    let b = 10;
    let x = UiChip::<BabyBear>::solve_lui(b);
    assert_eq!(x, [0, 0, 160, 0]);
}

fn prepare_lui_write_execute(
    tester: &mut MachineChipTestBuilder<F>,
    chip: &mut UiChip<F>,
    rng: &mut StdRng,
) {
    let address_range = || 0usize..1 << 29;

    let address_a = rng.gen_range(address_range()); // op_a
    let imm = rng.gen_range(address_range()); // op_b

    let x = UiChip::<F>::solve_lui(imm as u32);

    tester.execute(chip, Instruction::from_usize(Opcode::LUI, [address_a, imm]));
    assert_eq!(
        x.map(F::from_canonical_u32),
        tester.read::<4>(1usize, address_a)
    );
}

#[test]
fn lui_test() {
    let mut rng = create_seeded_rng();
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = UiChip::<F>::new(tester.execution_bus(), tester.memory_chip());
    let num_tests: usize = 1;

    for _ in 0..num_tests {
        prepare_lui_write_execute(&mut tester, &mut chip, &mut rng);
    }

    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}
