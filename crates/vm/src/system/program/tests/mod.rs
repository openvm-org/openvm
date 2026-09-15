use std::{iter, sync::Arc};

use openvm_circuit_primitives::Chip;
use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::{Program, DEFAULT_PC_STEP},
    LocalOpcode, VmOpcode,
};
use openvm_stark_backend::{
    any_air_arc_vec,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{AirProvingContext, CommittedTraceData, TraceCommitter},
    test_utils::dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    StarkEngine, StarkTestError,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::{
    arch::{instructions::SystemOpcode::*, testing::READ_INSTRUCTION_BUS},
    system::program::{trace::generate_cached_trace, ProgramAir, ProgramBus, ProgramChip},
    utils::test_cpu_engine,
};

// The tests do not need anything about opcodes besides the global opcode offset, so we use
// constants to avoid extra dependencies.
pub(crate) const LOADW: VmOpcode = VmOpcode::from_usize(0x210);
pub(crate) const STOREW: VmOpcode = VmOpcode::from_usize(0x213);
pub(crate) const BEQ: VmOpcode = VmOpcode::from_usize(0x220);
pub(crate) const SUB: VmOpcode = VmOpcode::from_usize(0x201);
pub(crate) const JAL: VmOpcode = VmOpcode::from_usize(0x230);
pub(crate) const BNE: VmOpcode = VmOpcode::from_usize(0x221);

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

    let engine = test_cpu_engine();
    let exe = VmExe::new(program);
    let cached_trace = generate_cached_trace(&exe.program);
    let (commitment, pcs_data) = TraceCommitter::commit(engine.device(), &[&cached_trace]).unwrap();
    let cached = CommittedTraceData {
        commitment,
        data: Arc::new(pcs_data),
        trace: cached_trace,
    };
    let chip = ProgramChip {
        filtered_exec_frequencies,
        cached: Some(cached),
        _marker: std::marker::PhantomData,
    };
    let ctx = chip.generate_proving_ctx(());

    let counter_air = DummyInteractionAir::new(9, true, bus.inner.index);
    let mut program_cells = vec![];
    for (index, frequency) in execution_frequencies.into_iter().enumerate() {
        let option = exe.program.get_instruction_and_debug_info(index);
        if let Some((instruction, _)) = option {
            program_cells.extend([
                BabyBear::from_u32(frequency),
                BabyBear::from_usize(index * (DEFAULT_PC_STEP as usize)),
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

    let counter_trace = RowMajorMatrix::new(program_cells, 10);
    println!("trace height = {original_height}");
    println!("counter trace height = {}", Matrix::height(&counter_trace));

    engine
        .run_test(
            any_air_arc_vec!(program_air, counter_air),
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
        Instruction::large_from_isize(STOREW, n, 0, 0, 0, 1, 0, 1),
        // word[1]_1 <- word[1]_1
        Instruction::large_from_isize(STOREW, 1, 1, 0, 0, 1, 0, 1),
        // if word[0]_1 == 0 then pc += 3*DEFAULT_PC_STEP
        Instruction::from_isize(BEQ, 0, 0, 3 * DEFAULT_PC_STEP as isize, 1, 0),
        // word[0]_1 <- word[0]_1 - word[1]_1
        Instruction::from_isize(SUB, 0, 0, 1, 1, 1),
        // word[2]_1 <- pc + DEFAULT_PC_STEP, pc -= 2*DEFAULT_PC_STEP
        Instruction::from_isize(JAL, 2, -2 * (DEFAULT_PC_STEP as isize), 0, 1, 0),
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
        Instruction::large_from_isize(STOREW, 5, 0, 0, 0, 1, 0, 1),
        // if word[0]_1 != 4 then pc += 3*DEFAULT_PC_STEP
        Instruction::from_isize(BNE, 0, 4, 3 * DEFAULT_PC_STEP as isize, 1, 0),
        // word[2]_1 <- pc + DEFAULT_PC_STEP, pc -= 2*DEFAULT_PC_STEP
        Instruction::from_isize(JAL, 2, -2 * DEFAULT_PC_STEP as isize, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        // if word[0]_1 == 5 then pc -= DEFAULT_PC_STEP
        Instruction::from_isize(BEQ, 0, 5, -(DEFAULT_PC_STEP as isize), 1, 0),
    ];

    let program = Program::from_instructions(&instructions);

    interaction_test(program, vec![0, 2, 4, 1]);
}

#[test]
fn test_program_negative() {
    let instructions = vec![
        Instruction::large_from_isize(STOREW, -1, 0, 0, 0, 1, 0, 1),
        Instruction::large_from_isize(LOADW, -1, 0, 0, 1, 1, 0, 1),
        Instruction::large_from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0, 0, 0),
    ];
    let bus = ProgramBus::new(READ_INSTRUCTION_BUS);
    let program = Program::from_instructions(&instructions);
    let program_air = ProgramAir::new(bus);

    let execution_frequencies = vec![1; instructions.len()];
    let engine = test_cpu_engine();
    let exe = VmExe::new(program);
    let cached_trace = generate_cached_trace(&exe.program);
    let (commitment, pcs_data) = TraceCommitter::commit(engine.device(), &[&cached_trace]).unwrap();
    let cached = CommittedTraceData {
        commitment,
        data: Arc::new(pcs_data),
        trace: cached_trace,
    };
    let chip = ProgramChip {
        filtered_exec_frequencies: execution_frequencies.clone(),
        cached: Some(cached),
        _marker: std::marker::PhantomData,
    };
    let ctx = chip.generate_proving_ctx(());

    let counter_air = DummyInteractionAir::new(7, true, bus.inner.index);
    let mut program_rows = vec![];
    for (pc_idx, instruction) in instructions.iter().enumerate() {
        program_rows.extend(vec![
            BabyBear::from_u32(execution_frequencies[pc_idx]),
            BabyBear::from_usize(pc_idx * DEFAULT_PC_STEP as usize),
            instruction.opcode.to_field(),
            instruction.a,
            instruction.b,
            instruction.c,
            instruction.d,
            instruction.e,
        ]);
    }
    let width = 8;
    let mut counter_trace = RowMajorMatrix::new(program_rows, width);
    counter_trace.row_mut(1)[1] = BabyBear::ZERO;
    let rows_used = Matrix::height(&counter_trace);
    let height = rows_used.next_power_of_two();
    counter_trace.values.resize(height * width, BabyBear::ZERO);
    let result = engine.run_test(
        any_air_arc_vec!(program_air, counter_air),
        vec![ctx, AirProvingContext::simple_no_pis(counter_trace)],
    );
    assert!(matches!(result, Err(StarkTestError::Prover(_))));
}

#[test]
fn test_program_with_undefined_instructions() {
    let n = 2;

    // see core/tests/mod.rs
    let instructions = vec![
        // word[0]_1 <- word[n]_0
        Some(Instruction::large_from_isize(STOREW, n, 0, 0, 0, 1, 0, 1)),
        // word[1]_1 <- word[1]_1
        Some(Instruction::large_from_isize(STOREW, 1, 1, 0, 0, 1, 0, 1)),
        // if word[0]_1 == n then pc += 3*DEFAULT_PC_STEP
        Some(Instruction::from_isize(
            BEQ,
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

fn check_boundary_columns(num_instructions: usize, corrupt: Option<(usize, usize)>) {
    let instruction = Instruction::from_usize(TERMINATE.global_opcode(), [0, 0, 0]);
    let program = Program::from_instructions(&vec![instruction; num_instructions]);
    let mut cached_trace = generate_cached_trace(&program);
    let height = cached_trace.height();
    assert_eq!(cached_trace.width(), 11);
    for row in 0..height {
        let values = cached_trace.row_slice(row).unwrap();
        assert_eq!(
            values[0],
            BabyBear::from_u32(1 + u32::from(row == height - 1))
        );
        assert_eq!(values[10], BabyBear::from_bool(row == 0));
    }
    if let Some((row, column)) = corrupt {
        cached_trace.row_mut(row)[column] += BabyBear::ONE;
    }

    let engine = test_cpu_engine();
    let (commitment, data) = TraceCommitter::commit(engine.device(), &[&cached_trace]).unwrap();
    let cached = CommittedTraceData {
        commitment,
        data: Arc::new(data),
        trace: cached_trace,
    };
    let chip = ProgramChip {
        filtered_exec_frequencies: vec![0; num_instructions],
        cached: Some(cached),
        _marker: std::marker::PhantomData,
    };
    let air = ProgramAir::new(ProgramBus::new(READ_INSTRUCTION_BUS));
    let result = engine.run_test(any_air_arc_vec!(air), vec![chip.generate_proving_ctx(())]);
    if corrupt.is_some() {
        assert!(
            result.is_err(),
            "corrupted boundary column must be rejected"
        );
    } else {
        result.expect("valid boundary columns must verify");
    }
}

#[test_case::test_case(0; "empty")]
#[test_case::test_case(1; "single_row")]
#[test_case::test_case(2; "two_rows")]
#[test_case::test_case(5; "padding")]
#[test_case::test_case(8; "no_padding")]
fn test_program_boundary_columns(num_instructions: usize) {
    check_boundary_columns(num_instructions, None);
}

#[test_case::test_case(5, 0; "first_row")]
#[test_case::test_case(5, 1; "interior_row_1")]
#[test_case::test_case(5, 2; "interior_row_2")]
#[test_case::test_case(5, 3; "interior_row_3")]
#[test_case::test_case(5, 4; "last_instruction")]
#[test_case::test_case(5, 5; "first_padding_row")]
#[test_case::test_case(5, 6; "interior_padding_row")]
#[test_case::test_case(5, 7; "last_padding_row")]
fn test_program_boundary_columns_negative(num_instructions: usize, row: usize) {
    // All frequencies are zero, including instruction and padding rows.
    for column in [0, 10] {
        check_boundary_columns(num_instructions, Some((row, column)));
    }
}

#[test]
fn test_program_single_row_boundary_columns_negative() {
    for column in [0, 10] {
        check_boundary_columns(1, Some((0, column)));
    }
}
