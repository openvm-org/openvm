use std::{iter, sync::Arc};

use afs_stark_backend::prover::{helper::AirProofInputTestHelper, types::AirProofInput};
use ax_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir, engine::StarkFriEngine,
};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::Program;
use crate::{
    arch::instructions::{CoreOpcode::*, FieldArithmeticOpcode::*, UsizeOpcode},
    kernels::core::READ_INSTRUCTION_BUS,
    system::program::{columns::ProgramCols, Instruction, ProgramChip},
};

#[test]
fn test_flatten_fromslice_roundtrip() {
    let num_cols = ProgramCols::<usize>::width();
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = ProgramCols::<usize>::from_slice(&all_cols);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

fn interaction_test(program: Program<BabyBear>, execution: Vec<u32>) {
    let instructions = program.instructions();
    let mut chip = ProgramChip::new(program);
    let mut execution_frequencies = vec![0; instructions.len()];
    for pc in execution {
        execution_frequencies[pc as usize] += 1;
        chip.get_instruction(pc).unwrap();
    }
    let air = chip.air.clone();
    let cached_trace = chip.generate_cached_trace();
    let main_trace = chip.generate_trace();

    let counter_air = DummyInteractionAir::new(9, true, READ_INSTRUCTION_BUS);
    let mut program_cells = vec![];
    for (pc, instruction) in instructions.iter().enumerate() {
        program_cells.extend(vec![
            BabyBear::from_canonical_usize(execution_frequencies[pc]),
            BabyBear::from_canonical_usize(pc),
            BabyBear::from_canonical_usize(instruction.opcode),
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
    let width = BaseAir::<BabyBear>::width(&air);
    let desired_height = instructions.len().next_power_of_two();
    let cells_to_add = (desired_height - instructions.len()) * width;
    program_cells.extend(iter::repeat(BabyBear::zero()).take(cells_to_add));

    let counter_trace = RowMajorMatrix::new(program_cells, 10);
    println!("trace height = {}", main_trace.height());
    println!("counter trace height = {}", counter_trace.height());

    BabyBearPoseidon2Engine::run_test_fast(vec![
        AirProofInput::cached_traces_no_pis(Arc::new(air), vec![cached_trace], main_trace),
        AirProofInput::simple_no_pis(Arc::new(counter_air), counter_trace),
    ])
    .expect("Verification failed");
}

#[test]
fn test_program_1() {
    let n = 2;

    // see core/tests/mod.rs
    let instructions = vec![
        // word[0]_1 <- word[n]_0
        Instruction::large_from_isize(STOREW.with_default_offset(), n, 0, 0, 0, 1, 0, 1),
        // word[1]_1 <- word[1]_1
        Instruction::large_from_isize(STOREW.with_default_offset(), 1, 1, 0, 0, 1, 0, 1),
        // if word[0]_1 == 0 then pc += 3
        Instruction::from_isize(BEQ.with_default_offset(), 0, 0, 3, 1, 0),
        // word[0]_1 <- word[0]_1 - word[1]_1
        Instruction::from_isize(SUB.with_default_offset(), 0, 0, 1, 1, 1),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::from_isize(JAL.with_default_offset(), 2, -2, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    interaction_test(program, vec![0, 3, 2, 5]);
}

#[test]
fn test_program_without_field_arithmetic() {
    // see core/tests/mod.rs
    let instructions = vec![
        // word[0]_1 <- word[5]_0
        Instruction::large_from_isize(STOREW.with_default_offset(), 5, 0, 0, 0, 1, 0, 1),
        // if word[0]_1 != 4 then pc += 2
        Instruction::from_isize(BNE.with_default_offset(), 0, 4, 3, 1, 0),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::from_isize(JAL.with_default_offset(), 2, -2, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0),
        // if word[0]_1 == 5 then pc -= 1
        Instruction::from_isize(BEQ.with_default_offset(), 0, 5, -1, 1, 0),
    ];

    let program = Program::from_instructions(&instructions);

    interaction_test(program, vec![0, 2, 4, 1]);
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn test_program_negative() {
    let instructions = vec![
        Instruction::large_from_isize(STOREW.with_default_offset(), -1, 0, 0, 0, 1, 0, 1),
        Instruction::large_from_isize(LOADW.with_default_offset(), -1, 0, 0, 1, 1, 0, 1),
        Instruction::large_from_isize(TERMINATE.with_default_offset(), 0, 0, 0, 0, 0, 0, 0),
    ];
    let program = Program::from_instructions(&instructions);

    let mut chip = ProgramChip::new(program);
    let execution_frequencies = vec![1; instructions.len()];
    for pc in 0..instructions.len() {
        chip.get_instruction(pc as u32).unwrap();
    }
    let air = chip.air.clone();
    let cached_trace = chip.generate_cached_trace();
    let common_trace = chip.generate_trace();

    let counter_air = DummyInteractionAir::new(7, true, READ_INSTRUCTION_BUS);
    let mut program_rows = vec![];
    for (pc, instruction) in instructions.iter().enumerate() {
        program_rows.extend(vec![
            BabyBear::from_canonical_usize(execution_frequencies[pc]),
            BabyBear::from_canonical_usize(pc),
            BabyBear::from_canonical_usize(instruction.opcode),
            instruction.op_a,
            instruction.op_b,
            instruction.op_c,
            instruction.d,
            instruction.e,
        ]);
    }
    let mut counter_trace = RowMajorMatrix::new(program_rows, 8);
    counter_trace.row_mut(1)[1] = BabyBear::zero();

    BabyBearPoseidon2Engine::run_test_fast(vec![
        AirProofInput::cached_traces_no_pis(Arc::new(air), vec![cached_trace], common_trace),
        AirProofInput::simple_no_pis(Arc::new(counter_air), counter_trace),
    ])
    .expect("Verification failed");
}
