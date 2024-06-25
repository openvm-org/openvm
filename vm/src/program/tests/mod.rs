use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

use crate::cpu::{CpuChip, /*ARITHMETIC_BUS, MEMORY_BUS,*/ READ_INSTRUCTION_BUS};

use crate::cpu::{trace::Instruction, OpCode::*};
use crate::program::columns::ProgramPreprocessedCols;

use super::ProgramAir;

#[test]
fn test_flatten_fromslice_roundtrip() {
    let num_cols = ProgramPreprocessedCols::<usize>::get_width();
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = ProgramPreprocessedCols::<usize>::from_slice(&all_cols);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

fn interaction_test(is_field_arithmetic_enabled: bool, program: Vec<Instruction<BabyBear>>) {
    let cpu_chip = CpuChip::new(is_field_arithmetic_enabled);
    let execution = cpu_chip.generate_program_execution(program.clone());

    let air = ProgramAir::new(program);
    let trace = air.generate_trace(&execution);

    let counter_air = DummyInteractionAir::new(7, true, READ_INSTRUCTION_BUS);
    let mut program_rows = vec![];
    for (pc, instruction) in execution.program.iter().enumerate() {
        program_rows.extend(vec![
            execution.execution_frequencies[pc],
            BabyBear::from_canonical_usize(pc),
            BabyBear::from_canonical_usize(instruction.opcode as usize),
            instruction.op_a,
            instruction.op_b,
            instruction.op_c,
            instruction.d,
            instruction.e,
        ]);
    }
    while !program_rows.len().is_power_of_two() {
        program_rows.push(BabyBear::zero());
    }
    let counter_trace = RowMajorMatrix::new(program_rows, 8);
    println!("trace height = {}", trace.height());
    println!("counter trace height = {}", counter_trace.height());

    run_simple_test_no_pis(vec![&air, &counter_air], vec![trace, counter_trace])
        .expect("Verification failed");
}

#[test]
fn test_program_1() {
    let n = 2;

    // see cpu/tests/mod.rs
    let program = vec![
        // word[0]_1 <- word[n]_0
        Instruction::from_isize(STOREW, n, 0, 0, 0, 1),
        // word[1]_1 <- word[1]_1
        Instruction::from_isize(STOREW, 1, 1, 0, 0, 1),
        // if word[0]_1 == 0 then pc += 3
        Instruction::from_isize(BEQ, 0, 0, 3, 1, 0),
        // word[0]_1 <- word[0]_1 - word[1]_1
        Instruction::from_isize(FSUB, 0, 0, 1, 1, 1),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::from_isize(JAL, 2, -2, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    interaction_test(true, program.clone());
}

#[test]
fn test_program_without_field_arithmetic() {
    let field_arithmetic_enabled = false;

    // see cpu/tests/mod.rs
    let program = vec![
        // word[0]_1 <- word[5]_0
        Instruction::from_isize(STOREW, 5, 0, 0, 0, 1),
        // if word[0]_1 != 4 then pc += 2
        Instruction::from_isize(BNE, 0, 4, 3, 1, 0),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::from_isize(JAL, 2, -2, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
        // if word[0]_1 == 5 then pc -= 1
        Instruction::from_isize(BEQ, 0, 5, -1, 1, 0),
    ];

    interaction_test(field_arithmetic_enabled, program.clone());
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn test_program_negative() {
    let program = vec![
        Instruction::from_isize(STOREW, -1, 0, 0, 0, 1),
        Instruction::from_isize(LOADW, -1, 0, 0, 1, 1),
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    let cpu_chip = CpuChip::new(true);
    let execution = cpu_chip.generate_program_execution(program.clone());

    let air = ProgramAir { program };
    let trace = air.generate_trace(&execution);

    let counter_air = DummyInteractionAir::new(7, true, READ_INSTRUCTION_BUS);
    let mut program_rows = vec![];
    for (pc, instruction) in execution.program.iter().enumerate() {
        program_rows.extend(vec![
            execution.execution_frequencies[pc],
            BabyBear::from_canonical_usize(pc),
            BabyBear::from_canonical_usize(instruction.opcode as usize),
            instruction.op_a,
            instruction.op_b,
            instruction.op_c,
            instruction.d,
            instruction.e,
        ]);
    }
    let mut counter_trace = RowMajorMatrix::new(program_rows, 8);
    counter_trace.row_mut(1)[1] = BabyBear::zero();

    run_simple_test_no_pis(vec![&air, &counter_air], vec![trace, counter_trace])
        .expect("Incorrect failure mode");
}
