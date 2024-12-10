use std::{iter, sync::Arc};

use ax_stark_backend::{
    p3_field::AbstractField,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::types::AirProofInput,
};
use ax_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        baby_bear_poseidon2_root::BabyBearPoseidon2RootConfig,
    },
    dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};
use axvm_instructions::{
    instruction::Instruction,
    program::{Program, DEFAULT_PC_STEP},
    AxVmOpcode,
};
use axvm_native_compiler::{
    FieldArithmeticOpcode::*, NativeBranchEqualOpcode, NativeJalOpcode::*, NativeLoadStoreOpcode::*,
};
use axvm_rv32im_transpiler::BranchEqualOpcode::*;
use serde::{de::DeserializeOwned, Serialize};
use static_assertions::assert_impl_all;

use crate::{
    arch::{instructions::SystemOpcode::*, READ_INSTRUCTION_BUS},
    system::program::{trace::AxVmCommittedExe, ProgramBus, ProgramChip},
};

assert_impl_all!(AxVmCommittedExe<BabyBearPoseidon2Config>: Serialize, DeserializeOwned);
assert_impl_all!(AxVmCommittedExe<BabyBearPoseidon2RootConfig>: Serialize, DeserializeOwned);

fn interaction_test(program: Program<BabyBear>, execution: Vec<u32>) {
    let instructions = program.instructions();
    let bus = ProgramBus(READ_INSTRUCTION_BUS);
    let mut chip = ProgramChip::new_with_program(program, bus);
    let mut execution_frequencies = vec![0; instructions.len()];
    for pc_idx in execution {
        execution_frequencies[pc_idx as usize] += 1;
        chip.get_instruction(pc_idx * DEFAULT_PC_STEP).unwrap();
    }
    let program_proof_input = chip.generate_air_proof_input(None);

    let counter_air = DummyInteractionAir::new(9, true, bus.0);
    let mut program_cells = vec![];
    for (pc_idx, instruction) in instructions.iter().enumerate() {
        program_cells.extend(vec![
            BabyBear::from_canonical_usize(execution_frequencies[pc_idx]), // hacky: we should switch execution_frequencies into hashmap
            BabyBear::from_canonical_usize(pc_idx * (DEFAULT_PC_STEP as usize)),
            instruction.opcode.to_field(),
            instruction.a,
            instruction.b,
            instruction.c,
            instruction.d,
            instruction.e,
            instruction.f,
            instruction.g,
        ]);
    }

    // Pad program cells with zeroes to make height a power of two.
    let width = 10;
    let desired_height = instructions.len().next_power_of_two();
    let cells_to_add = (desired_height - instructions.len()) * width;
    program_cells.extend(iter::repeat(BabyBear::ZERO).take(cells_to_add));

    let counter_trace = RowMajorMatrix::new(program_cells, 10);
    println!("trace height = {}", instructions.len());
    println!("counter trace height = {}", counter_trace.height());

    BabyBearPoseidon2Engine::run_test_fast(vec![
        program_proof_input,
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
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(STOREW), n, 0, 0, 0, 1, 0, 1),
        // word[1]_1 <- word[1]_1
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(STOREW), 1, 1, 0, 0, 1, 0, 1),
        // if word[0]_1 == 0 then pc += 3*DEFAULT_PC_STEP
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BEQ)),
            0,
            0,
            3 * DEFAULT_PC_STEP as isize,
            1,
            0,
        ),
        // word[0]_1 <- word[0]_1 - word[1]_1
        Instruction::from_isize(AxVmOpcode::with_default_offset(SUB), 0, 0, 1, 1, 1),
        // word[2]_1 <- pc + DEFAULT_PC_STEP, pc -= 2*DEFAULT_PC_STEP
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(JAL),
            2,
            -2 * (DEFAULT_PC_STEP as isize),
            0,
            1,
            0,
        ),
        // terminate
        Instruction::from_isize(AxVmOpcode::with_default_offset(TERMINATE), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    interaction_test(program, vec![0, 3, 2, 5]);
}

#[test]
fn test_program_without_field_arithmetic() {
    // see core/tests/mod.rs
    let instructions = vec![
        // word[0]_1 <- word[5]_0
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(STOREW), 5, 0, 0, 0, 1, 0, 1),
        // if word[0]_1 != 4 then pc += 3*DEFAULT_PC_STEP
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BNE)),
            0,
            4,
            3 * DEFAULT_PC_STEP as isize,
            1,
            0,
        ),
        // word[2]_1 <- pc + DEFAULT_PC_STEP, pc -= 2*DEFAULT_PC_STEP
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(JAL),
            2,
            -2 * DEFAULT_PC_STEP as isize,
            0,
            1,
            0,
        ),
        // terminate
        Instruction::from_isize(AxVmOpcode::with_default_offset(TERMINATE), 0, 0, 0, 0, 0),
        // if word[0]_1 == 5 then pc -= DEFAULT_PC_STEP
        Instruction::from_isize(
            AxVmOpcode::with_default_offset(NativeBranchEqualOpcode(BEQ)),
            0,
            5,
            -(DEFAULT_PC_STEP as isize),
            1,
            0,
        ),
    ];

    let program = Program::from_instructions(&instructions);

    interaction_test(program, vec![0, 2, 4, 1]);
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn test_program_negative() {
    let instructions = vec![
        Instruction::large_from_isize(
            AxVmOpcode::with_default_offset(STOREW),
            -1,
            0,
            0,
            0,
            1,
            0,
            1,
        ),
        Instruction::large_from_isize(AxVmOpcode::with_default_offset(LOADW), -1, 0, 0, 1, 1, 0, 1),
        Instruction::large_from_isize(
            AxVmOpcode::with_default_offset(TERMINATE),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
    ];
    let bus = ProgramBus(READ_INSTRUCTION_BUS);
    let program = Program::from_instructions(&instructions);

    let mut chip = ProgramChip::new_with_program(program, bus);
    let execution_frequencies = vec![1; instructions.len()];
    for pc_idx in 0..instructions.len() {
        chip.get_instruction(pc_idx as u32 * DEFAULT_PC_STEP)
            .unwrap();
    }
    let program_proof_input = chip.generate_air_proof_input(None);

    let counter_air = DummyInteractionAir::new(7, true, bus.0);
    let mut program_rows = vec![];
    for (pc_idx, instruction) in instructions.iter().enumerate() {
        program_rows.extend(vec![
            BabyBear::from_canonical_usize(execution_frequencies[pc_idx]),
            BabyBear::from_canonical_usize(pc_idx * DEFAULT_PC_STEP as usize),
            instruction.opcode.to_field(),
            instruction.a,
            instruction.b,
            instruction.c,
            instruction.d,
            instruction.e,
        ]);
    }
    let mut counter_trace = RowMajorMatrix::new(program_rows, 8);
    counter_trace.row_mut(1)[1] = BabyBear::ZERO;

    BabyBearPoseidon2Engine::run_test_fast(vec![
        program_proof_input,
        AirProofInput::simple_no_pis(Arc::new(counter_air), counter_trace),
    ])
    .expect("Verification failed");
}
