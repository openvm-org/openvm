use std::{iter, sync::Arc};

use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::{Program, DEFAULT_PC_STEP},
    LocalOpcode,
};
use openvm_native_compiler::{
    FieldArithmeticOpcode::*, NativeBranchEqualOpcode, NativeJalOpcode::*, NativeLoadStoreOpcode::*,
};
use openvm_rv32im_transpiler::BranchEqualOpcode::*;
use openvm_stark_backend::{
    config::StarkGenericConfig,
    engine::StarkEngine,
    p3_field::FieldAlgebra,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::types::AirProvingContext,
    Chip,
};
use openvm_stark_sdk::{
    any_rap_arc_vec,
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        baby_bear_poseidon2_root::BabyBearPoseidon2RootConfig,
        FriParameters,
    },
    dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};
use serde::{de::DeserializeOwned, Serialize};
use static_assertions::assert_impl_all;

use crate::{
    arch::{instructions::SystemOpcode::*, testing::READ_INSTRUCTION_BUS},
    system::program::{trace::VmCommittedExe, ProgramAir, ProgramBus, ProgramChip},
};

assert_impl_all!(VmCommittedExe<BabyBearPoseidon2Config>: Serialize, DeserializeOwned);
assert_impl_all!(VmCommittedExe<BabyBearPoseidon2RootConfig>: Serialize, DeserializeOwned);

fn interaction_test(program: Program<BabyBear>, execution: Vec<u32>) {
    let mut execution_frequencies = vec![0; program.len()];
    for pc_idx in execution {
        execution_frequencies[pc_idx as usize] += 1;
    }
    let filtered_exec_frequencies: Vec<_> = program
        .instructions_and_debug_infos
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.is_some())
        .map(|(i, _)| execution_frequencies[i])
        .collect();
    let original_height = filtered_exec_frequencies.len();

    let bus = ProgramBus::new(READ_INSTRUCTION_BUS);
    let program_air = ProgramAir::new(bus);

    let engine = BabyBearPoseidon2Engine::new(FriParameters::new_for_testing(1));
    let exe = VmExe::new(program);
    let committed_exe =
        VmCommittedExe::<BabyBearPoseidon2Config>::commit(exe, engine.config().pcs());
    let cached = committed_exe.get_committed_trace();
    let chip = ProgramChip {
        filtered_exec_frequencies,
        cached: Some(cached),
    };
    let ctx = chip.generate_proving_ctx(());

    let counter_air = DummyInteractionAir::new(9, true, bus.inner.index);
    let mut program_cells = vec![];
    let program = &committed_exe.exe.program;
    for (index, frequency) in execution_frequencies.into_iter().enumerate() {
        let option = program.get_instruction_and_debug_info(index);
        if let Some((instruction, _)) = option {
            program_cells.extend([
                BabyBear::from_canonical_u32(frequency),
                BabyBear::from_canonical_usize(index * (DEFAULT_PC_STEP as usize)),
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
    }

    // Pad program cells with zeroes to make height a power of two.
    let width = 10;
    let desired_height = original_height.next_power_of_two();
    let cells_to_add = (desired_height - original_height) * width;
    program_cells.extend(iter::repeat_n(BabyBear::ZERO, cells_to_add));

    let counter_trace = Arc::new(RowMajorMatrix::new(program_cells, 10));
    println!("trace height = {}", original_height);
    println!("counter trace height = {}", counter_trace.height());

    engine
        .run_test(
            any_rap_arc_vec!(program_air, counter_air),
            vec![ctx, AirProvingContext::simple_no_pis(counter_trace)],
        )
        .expect("Verification failed");
}

#[test]
fn test_program_1() {
    let n = 2;

    // see core/tests/mod.rs
    let instructions = vec![
        // word[0]_1 <- word[n]_0
        Instruction::large_from_isize(STOREW.global_opcode(), n, 0, 0, 0, 1, 0, 1),
        // word[1]_1 <- word[1]_1
        Instruction::large_from_isize(STOREW.global_opcode(), 1, 1, 0, 0, 1, 0, 1),
        // if word[0]_1 == 0 then pc += 3*DEFAULT_PC_STEP
        Instruction::from_isize(
            NativeBranchEqualOpcode(BEQ).global_opcode(),
            0,
            0,
            3 * DEFAULT_PC_STEP as isize,
            1,
            0,
        ),
        // word[0]_1 <- word[0]_1 - word[1]_1
        Instruction::from_isize(SUB.global_opcode(), 0, 0, 1, 1, 1),
        // word[2]_1 <- pc + DEFAULT_PC_STEP, pc -= 2*DEFAULT_PC_STEP
        Instruction::from_isize(
            JAL.global_opcode(),
            2,
            -2 * (DEFAULT_PC_STEP as isize),
            0,
            1,
            0,
        ),
        // terminate
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let program = Program::from_instructions(&instructions);

    interaction_test(program, vec![0, 3, 2, 5]);
}

#[test]
fn test_program_without_field_arithmetic() {
    // see core/tests/mod.rs
    let instructions = vec![
        // word[0]_1 <- word[5]_0
        Instruction::large_from_isize(STOREW.global_opcode(), 5, 0, 0, 0, 1, 0, 1),
        // if word[0]_1 != 4 then pc += 3*DEFAULT_PC_STEP
        Instruction::from_isize(
            NativeBranchEqualOpcode(BNE).global_opcode(),
            0,
            4,
            3 * DEFAULT_PC_STEP as isize,
            1,
            0,
        ),
        // word[2]_1 <- pc + DEFAULT_PC_STEP, pc -= 2*DEFAULT_PC_STEP
        Instruction::from_isize(
            JAL.global_opcode(),
            2,
            -2 * DEFAULT_PC_STEP as isize,
            0,
            1,
            0,
        ),
        // terminate
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        // if word[0]_1 == 5 then pc -= DEFAULT_PC_STEP
        Instruction::from_isize(
            NativeBranchEqualOpcode(BEQ).global_opcode(),
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
#[should_panic(expected = "LogUp multiset equality check failed.")]
fn test_program_negative() {
    let instructions = vec![
        Instruction::large_from_isize(STOREW.global_opcode(), -1, 0, 0, 0, 1, 0, 1),
        Instruction::large_from_isize(LOADW.global_opcode(), -1, 0, 0, 1, 1, 0, 1),
        Instruction::large_from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0, 0, 0),
    ];
    let bus = ProgramBus::new(READ_INSTRUCTION_BUS);
    let program = Program::from_instructions(&instructions);
    let program_air = ProgramAir::new(bus);

    let execution_frequencies = vec![1; instructions.len()];
    let engine = BabyBearPoseidon2Engine::new(FriParameters::new_for_testing(1));
    let exe = VmExe::new(program);
    let committed_exe =
        VmCommittedExe::<BabyBearPoseidon2Config>::commit(exe, engine.config().pcs());
    let cached = committed_exe.get_committed_trace();
    let chip = ProgramChip {
        filtered_exec_frequencies: execution_frequencies.clone(),
        cached: Some(cached),
    };
    let ctx = chip.generate_proving_ctx(());

    let counter_air = DummyInteractionAir::new(7, true, bus.inner.index);
    let mut program_rows = vec![];
    for (pc_idx, instruction) in instructions.iter().enumerate() {
        program_rows.extend(vec![
            BabyBear::from_canonical_u32(execution_frequencies[pc_idx]),
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
    let counter_trace = Arc::new(counter_trace);

    engine
        .run_test(
            any_rap_arc_vec!(program_air, counter_air),
            vec![ctx, AirProvingContext::simple_no_pis(counter_trace)],
        )
        .expect("Verification failed");
}

#[test]
fn test_program_with_undefined_instructions() {
    let n = 2;

    // see core/tests/mod.rs
    let instructions = vec![
        // word[0]_1 <- word[n]_0
        Some(Instruction::large_from_isize(
            STOREW.global_opcode(),
            n,
            0,
            0,
            0,
            1,
            0,
            1,
        )),
        // word[1]_1 <- word[1]_1
        Some(Instruction::large_from_isize(
            STOREW.global_opcode(),
            1,
            1,
            0,
            0,
            1,
            0,
            1,
        )),
        // if word[0]_1 == n then pc += 3*DEFAULT_PC_STEP
        Some(Instruction::from_isize(
            NativeBranchEqualOpcode(BEQ).global_opcode(),
            0,
            n,
            3 * DEFAULT_PC_STEP as isize,
            1,
            0,
        )),
        None,
        None,
        // terminate
        Some(Instruction::from_isize(
            TERMINATE.global_opcode(),
            0,
            0,
            0,
            0,
            0,
        )),
    ];

    let program = Program::new_without_debug_infos_with_option(&instructions, 0);

    interaction_test(program, vec![0, 2, 5]);
}
