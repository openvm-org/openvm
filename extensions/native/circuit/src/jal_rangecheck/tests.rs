use std::borrow::BorrowMut;

use openvm_circuit::arch::testing::{memory::gen_pointer, VmChipTestBuilder};
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    LocalOpcode, VmOpcode,
};
use openvm_native_compiler::{
    conversion::AS, NativeJalOpcode::*, NativeRangeCheckOpcode::RANGE_CHECK,
};
use openvm_stark_backend::{
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;

use super::{JalRangeCheckAir, JalRangeCheckStep};
use crate::{
    jal_rangecheck::{JalRangeCheckChip, JalRangeCheckCols},
    test_utils::write_native_array,
};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

fn create_test_chip(tester: &VmChipTestBuilder<F>) -> JalRangeCheckChip<F> {
    let mut chip = JalRangeCheckChip::<F>::new(
        JalRangeCheckAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            tester.range_checker().bus(),
        ),
        JalRangeCheckStep::new(tester.range_checker().clone()),
        tester.memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);

    chip
}

// `a_val` and `c` will be disregarded if opcode is JAL
fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut JalRangeCheckChip<F>,
    rng: &mut StdRng,
    opcode: VmOpcode,
    a_val: Option<u32>,
    b: Option<u32>,
    c: Option<u32>,
) {
    if opcode == JAL.global_opcode() {
        let initial_pc = rng.gen_range(0..(1 << PC_BITS));
        let a = gen_pointer(rng, 1);
        let final_pc = F::from_canonical_u32(rng.gen_range(0..(1 << PC_BITS)));
        let b = b.unwrap_or((final_pc - F::from_canonical_u32(initial_pc)).as_canonical_u32());
        tester.execute_with_pc(
            chip,
            &Instruction::from_usize(opcode, [a, b as usize, 0, AS::Native as usize, 0, 0, 0]),
            initial_pc,
        );

        let final_pc = tester.execution.last_to_pc();
        let expected_final_pc = F::from_canonical_u32(initial_pc) + F::from_canonical_u32(b);
        assert_eq!(final_pc, expected_final_pc);
        let result_a_val = tester.read::<1>(AS::Native as usize, a)[0].as_canonical_u32();
        let expected_a_val = initial_pc + DEFAULT_PC_STEP;
        assert_eq!(result_a_val, expected_a_val);
    } else {
        let a_val = a_val.unwrap_or(rng.gen_range(0..(1 << 30)));
        let a = write_native_array(tester, rng, Some([F::from_canonical_u32(a_val)])).1;
        let x = a_val & 0xffff;
        let y = a_val >> 16;

        let min_b = 32 - x.leading_zeros();
        let min_c = 32 - y.leading_zeros();
        let b = b.unwrap_or(rng.gen_range(min_b..=16));
        let c = c.unwrap_or(rng.gen_range(min_c..=14));
        tester.execute(
            chip,
            &Instruction::from_usize(
                opcode,
                [a, b as usize, c as usize, AS::Native as usize, 0, 0, 0],
            ),
        );
        // There is nothing to assert for range check since it doesn't write to the memory
    };
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[test_case(JAL.global_opcode(), 100)]
#[test_case(RANGE_CHECK.global_opcode(), 100)]
fn rand_jal_range_check_test(opcode: VmOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut chip = create_test_chip(&tester);

    for _ in 0..num_ops {
        set_and_execute(&mut tester, &mut chip, &mut rng, opcode, None, None, None);
    }
    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn range_check_edge_cases_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut chip = create_test_chip(&tester);

    set_and_execute(
        &mut tester,
        &mut chip,
        &mut rng,
        RANGE_CHECK.global_opcode(),
        Some(0),
        None,
        None,
    );
    set_and_execute(
        &mut tester,
        &mut chip,
        &mut rng,
        RANGE_CHECK.global_opcode(),
        Some((1 << 30) - 1),
        None,
        None,
    );

    // x = 0
    let a = rng.gen_range(0..(1 << 14)) << 16;
    set_and_execute(
        &mut tester,
        &mut chip,
        &mut rng,
        RANGE_CHECK.global_opcode(),
        Some(a),
        None,
        None,
    );

    // y = 0
    let a = rng.gen_range(0..(1 << 16));
    set_and_execute(
        &mut tester,
        &mut chip,
        &mut rng,
        RANGE_CHECK.global_opcode(),
        Some(a),
        None,
        None,
    );

    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default)]
struct JalRangeCheckPrankValues {
    pub flags: Option<[bool; 2]>,
    pub a_val: Option<u32>,
    pub b: Option<u32>,
    pub c: Option<u32>,
    pub y: Option<u32>,
}

fn run_negative_jal_range_check_test(
    opcode: VmOpcode,
    a_val: Option<u32>,
    b: Option<u32>,
    c: Option<u32>,
    prank_vals: JalRangeCheckPrankValues,
    error: VerificationError,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut chip = create_test_chip(&tester);
    set_and_execute(&mut tester, &mut chip, &mut rng, opcode, a_val, b, c);

    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut JalRangeCheckCols<F> = values[..].borrow_mut();

        if let Some(flags) = prank_vals.flags {
            cols.is_jal = F::from_bool(flags[0]);
            cols.is_range_check = F::from_bool(flags[1]);
        }
        if let Some(a_val) = prank_vals.a_val {
            cols.writes_aux
                .set_prev_data([F::from_canonical_u32(a_val)]);
        }

        if let Some(b) = prank_vals.b {
            cols.b = F::from_canonical_u32(b);
        }
        if let Some(c) = prank_vals.c {
            cols.c = F::from_canonical_u32(c);
        }
        if let Some(y) = prank_vals.y {
            cols.y = F::from_canonical_u32(y);
        }

        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(chip, modify_trace)
        .finalize();
    tester.simple_test_with_expected_error(error);
}

#[test]
fn negative_range_check_test() {
    run_negative_jal_range_check_test(
        RANGE_CHECK.global_opcode(),
        Some(2),
        Some(2),
        Some(1),
        JalRangeCheckPrankValues {
            b: Some(1),
            ..Default::default()
        },
        VerificationError::ChallengePhaseError,
    );
    run_negative_jal_range_check_test(
        RANGE_CHECK.global_opcode(),
        Some(1 << 16),
        None,
        None,
        JalRangeCheckPrankValues {
            c: Some(0),
            ..Default::default()
        },
        VerificationError::ChallengePhaseError,
    );
    run_negative_jal_range_check_test(
        RANGE_CHECK.global_opcode(),
        Some((1 << 30) - 1),
        None,
        None,
        JalRangeCheckPrankValues {
            a_val: Some(1 << 30),
            ..Default::default()
        },
        VerificationError::ChallengePhaseError,
    );
    run_negative_jal_range_check_test(
        RANGE_CHECK.global_opcode(),
        Some(1 << 17),
        None,
        None,
        JalRangeCheckPrankValues {
            y: Some(1),
            ..Default::default()
        },
        VerificationError::ChallengePhaseError,
    );
}

#[test]
fn negative_jal_test() {
    run_negative_jal_range_check_test(
        JAL.global_opcode(),
        None,
        None,
        None,
        JalRangeCheckPrankValues {
            b: Some(0),
            ..Default::default()
        },
        VerificationError::ChallengePhaseError,
    );
}
