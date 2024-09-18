use std::iter;

use ax_sdk::{
    any_rap_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
    interaction::dummy_interaction_air::DummyInteractionAir,
};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::Program;
use crate::{
    arch::{chips::MachineChip, instructions::Opcode::*},
    cpu::{trace::Instruction, READ_INSTRUCTION_BUS},
    program::{columns::ProgramPreprocessedCols, ProgramChip},
};

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

fn interaction_test(program: Program<BabyBear>, execution: Vec<usize>) {
    let instructions = program.instructions.clone();
    let mut chip = ProgramChip::new(program);
    let mut execution_frequencies = vec![0; instructions.len()];
    for pc in execution {
        execution_frequencies[pc] += 1;
        chip.get_instruction(pc).unwrap();
    }
    let air = chip.air.clone();
    let trace = chip.generate_trace();

    let counter_air = DummyInteractionAir::new(9, true, READ_INSTRUCTION_BUS);
    let mut program_cells = vec![];
    for (pc, instruction) in instructions.iter().enumerate() {
        program_cells.extend(vec![
            BabyBear::from_canonical_usize(execution_frequencies[pc]),
            BabyBear::from_canonical_usize(pc),
            BabyBear::from_canonical_usize(instruction.opcode as usize),
            instruction.op_a,
            instruction.op_b,
            instruction.op_c,
            instruction.d,
            instruction.e,
            instruction.op_f,
            instruction.op_g,
        ]);
    }

    // Pad program cells with zeroes to make height a power of two.
    let width = air.width() + ProgramPreprocessedCols::<BabyBear>::get_width();
    let desired_height = instructions.len().next_power_of_two();
    let cells_to_add = desired_height * width - instructions.len() * width;
    program_cells.extend(iter::repeat(BabyBear::zero()).take(cells_to_add));

    let counter_trace = RowMajorMatrix::new(program_cells, 10);
    println!("trace height = {}", trace.height());
    println!("counter trace height = {}", counter_trace.height());

    BabyBearPoseidon2Engine::run_simple_test_no_pis(
        &any_rap_vec![&air, &counter_air],
        vec![trace, counter_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_program_1() {
    let n = 2;

    // see cpu/tests/mod.rs
    let instructions = vec![
        // word[0]_1 <- word[n]_0
        Instruction::large_from_isize(STOREW, n, 0, 0, 0, 1, 0, 1),
        // word[1]_1 <- word[1]_1
        Instruction::large_from_isize(STOREW, 1, 1, 0, 0, 1, 0, 1),
        // if word[0]_1 == 0 then pc += 3
        Instruction::from_isize(BEQ, 0, 0, 3, 1, 0),
        // word[0]_1 <- word[0]_1 - word[1]_1
        Instruction::from_isize(FSUB, 0, 0, 1, 1, 1),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::from_isize(JAL, 2, -2, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    let program = Program {
        instructions,
        debug_infos: vec![None; 6],
    };

    interaction_test(program, vec![0, 3, 2, 5]);
}

#[test]
fn test_program_without_field_arithmetic() {
    // see cpu/tests/mod.rs
    let instructions = vec![
        // word[0]_1 <- word[5]_0
        Instruction::large_from_isize(STOREW, 5, 0, 0, 0, 1, 0, 1),
        // if word[0]_1 != 4 then pc += 2
        Instruction::from_isize(BNE, 0, 4, 3, 1, 0),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::from_isize(JAL, 2, -2, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
        // if word[0]_1 == 5 then pc -= 1
        Instruction::from_isize(BEQ, 0, 5, -1, 1, 0),
    ];

    let program = Program {
        instructions,
        debug_infos: vec![None; 5],
    };

    interaction_test(program, vec![0, 2, 4, 1]);
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn test_program_negative() {
    let instructions = vec![
        Instruction::large_from_isize(STOREW, -1, 0, 0, 0, 1, 0, 1),
        Instruction::large_from_isize(LOADW, -1, 0, 0, 1, 1, 0, 1),
        Instruction::large_from_isize(TERMINATE, 0, 0, 0, 0, 0, 0, 0),
    ];
    let program = Program {
        instructions: instructions.clone(),
        debug_infos: vec![None; 3],
    };

    let mut chip = ProgramChip::new(program);
    let execution_frequencies = vec![1; instructions.len()];
    for pc in 0..instructions.len() {
        chip.get_instruction(pc).unwrap();
    }
    let air = chip.air.clone();
    let trace = chip.generate_trace();

    let counter_air = DummyInteractionAir::new(7, true, READ_INSTRUCTION_BUS);
    let mut program_rows = vec![];
    for (pc, instruction) in instructions.iter().enumerate() {
        program_rows.extend(vec![
            BabyBear::from_canonical_usize(execution_frequencies[pc]),
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

    BabyBearPoseidon2Engine::run_simple_test_no_pis(
        &any_rap_vec![&air, &counter_air],
        vec![trace, counter_trace],
    )
    .expect("Incorrect failure mode");
}
