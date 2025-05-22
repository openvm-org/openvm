use std::borrow::BorrowMut;

use openvm_circuit::arch::testing::VmChipTestBuilder;
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    LocalOpcode,
};
use openvm_native_compiler::{NativeJalOpcode::*, NativeRangeCheckOpcode::RANGE_CHECK};
use openvm_stark_backend::{
    p3_field::{FieldAlgebra, PrimeField32},
    utils::disable_debug_builder,
    verifier::VerificationError,
    Chip,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use super::{JalRangeCheckAir, JalRangeCheckStep};
use crate::jal::{JalRangeCheckChip, JalRangeCheckCols};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

fn create_test_chip(tester: &VmChipTestBuilder<F>) -> JalRangeCheckChip<F> {
    JalRangeCheckChip::<F>::new(
        JalRangeCheckAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            tester.range_checker().bus(),
        ),
        JalRangeCheckStep::new(tester.range_checker().clone()),
        MAX_INS_CAPACITY,
        tester.memory_helper(),
    )
}

fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut JalRangeCheckChip<F>,
    rng: &mut StdRng,
    initial_imm: Option<u32>,
    initial_pc: Option<u32>,
) {
    let imm = initial_imm.unwrap_or(rng.gen_range(0..20));
    let a = rng.gen_range(0..32) << 2;
    let d = 4usize;

    tester.execute_with_pc(
        chip,
        &Instruction::from_usize(JAL.global_opcode(), [a, imm as usize, 0, d, 0, 0, 0]),
        initial_pc.unwrap_or(rng.gen_range(0..(1 << PC_BITS))),
    );
    let initial_pc = tester.execution.last_from_pc().as_canonical_u32();
    let final_pc = tester.execution.last_to_pc().as_canonical_u32();

    let next_pc = initial_pc + imm;
    let rd_data = initial_pc + DEFAULT_PC_STEP;

    assert_eq!(next_pc, final_pc);
    assert_eq!(rd_data, tester.read::<1>(d, a)[0].as_canonical_u32());
}

struct RangeCheckTestCase {
    val: u32,
    x_bit: u32,
    y_bit: u32,
}

fn set_and_execute_range_check(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut JalRangeCheckChip<F>,
    rng: &mut StdRng,
    test_cases: Vec<RangeCheckTestCase>,
) {
    let a = rng.gen_range(0..32) << 2;
    for RangeCheckTestCase { val, x_bit, y_bit } in test_cases {
        let d = 4usize;

        tester.write(d, a, [F::from_canonical_u32(val)]);
        tester.execute_with_pc(
            chip,
            &Instruction::from_usize(
                RANGE_CHECK.global_opcode(),
                [a, x_bit as usize, y_bit as usize, d, 0, 0, 0],
            ),
            rng.gen_range(0..(1 << PC_BITS)),
        );
    }
}

#[test]
fn rand_jal_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut chip = create_test_chip(&tester);

    let num_tests: usize = 100;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, None, None);
    }

    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rand_range_check_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut chip = create_test_chip(&tester);

    let f = |x: u32, y: u32| RangeCheckTestCase {
        val: x + y * (1 << 16),
        x_bit: 32 - x.leading_zeros(),
        y_bit: 32 - y.leading_zeros(),
    };
    let mut test_cases: Vec<_> = (0..10)
        .map(|_| {
            let x = 0;
            let y = rng.gen_range(0..1 << 14);
            f(x, y)
        })
        .collect();
    test_cases.extend((0..10).map(|_| {
        let x = rng.gen_range(0..1 << 16);
        let y = 0;
        f(x, y)
    }));
    test_cases.extend((0..10).map(|_| {
        let x = rng.gen_range(0..1 << 16);
        let y = rng.gen_range(0..1 << 14);
        f(x, y)
    }));
    test_cases.push(f((1 << 16) - 1, (1 << 14) - 1));
    set_and_execute_range_check(&mut tester, &mut chip, &mut rng, test_cases);
    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn negative_range_check_test() {
    {
        let mut rng = create_seeded_rng();
        let mut tester = VmChipTestBuilder::default();
        let mut chip = create_test_chip(&tester);
        chip.step.set_debug();

        set_and_execute_range_check(
            &mut tester,
            &mut chip,
            &mut rng,
            vec![RangeCheckTestCase {
                x_bit: 1,
                y_bit: 1,
                val: 2,
            }],
        );
        let tester = tester.build().load(chip).finalize();
        disable_debug_builder();
        let result = tester.simple_test();
        assert!(result.is_err());
    }
    {
        let mut rng = create_seeded_rng();
        let mut tester = VmChipTestBuilder::default();
        let mut chip = create_test_chip(&tester);
        chip.step.set_debug();

        set_and_execute_range_check(
            &mut tester,
            &mut chip,
            &mut rng,
            vec![RangeCheckTestCase {
                x_bit: 1,
                y_bit: 0,
                val: 1 << 16,
            }],
        );
        let tester = tester.build().load(chip).finalize();
        disable_debug_builder();
        let result = tester.simple_test();
        assert!(result.is_err());
    }
}

#[test]
fn negative_jal_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut chip = create_test_chip(&tester);

    set_and_execute(&mut tester, &mut chip, &mut rng, None, None);

    let tester = tester.build();

    let chip_air = chip.air();
    let mut chip_input = chip.generate_air_proof_input();
    let jal_trace = chip_input.raw.common_main.as_mut().unwrap();
    {
        let col: &mut JalRangeCheckCols<_> = jal_trace.row_mut(0).borrow_mut();
        col.b = F::from_canonical_u32(rng.gen_range(1 << 11..1 << 12));
    }
    disable_debug_builder();
    let tester = tester
        .load_air_proof_input((chip_air, chip_input))
        .finalize();
    let msg = format!(
        "Expected verification to fail with {:?}, but it didn't",
        VerificationError::ChallengePhaseError
    );
    let result = tester.simple_test();
    assert_eq!(
        result.err(),
        Some(VerificationError::ChallengePhaseError),
        "{}",
        msg
    );
}
