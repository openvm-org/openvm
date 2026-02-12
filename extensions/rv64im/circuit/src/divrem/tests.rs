use openvm_circuit::{
    arch::{execution_mode::ExecutionCtx, VmExecState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
    VmOpcode,
};
use openvm_rv64im_transpiler::Rv64DivRemOpcode;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use crate::{
    test_utils::{create_exec_state, execute_instruction, read_reg, write_reg},
    Rv64DivRemExecutor,
};

type F = BabyBear;

const REG_A: u32 = 0;
const REG_B: u32 = 8;
const REG_C: u32 = 16;
const START_PC: u32 = 0x1000;

fn make_instruction(opcode: Rv64DivRemOpcode, rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
    Instruction::from_usize::<7>(
        VmOpcode::from_usize(opcode.global_opcode_usize()),
        [
            rd as usize,
            rs1 as usize,
            rs2 as usize,
            RV32_REGISTER_AS as usize,
            RV32_REGISTER_AS as usize,
            0,
            0,
        ],
    )
}

fn execute(
    executor: &Rv64DivRemExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64DivRemOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}

// ---------------------------------------------------------------------------
// DIV (signed)
// ---------------------------------------------------------------------------

#[test]
fn test_div_basic() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 7);
    let inst = make_instruction(Rv64DivRemOpcode::DIV, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 6);
}

#[test]
fn test_div_by_zero() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 0);
    let inst = make_instruction(Rv64DivRemOpcode::DIV, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), u64::MAX);
}

#[test]
fn test_div_overflow() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // i64::MIN / -1 = i64::MIN (overflow case)
    write_reg(&mut state, REG_B, i64::MIN as u64);
    write_reg(&mut state, REG_C, u64::MAX); // -1
    let inst = make_instruction(Rv64DivRemOpcode::DIV, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), i64::MIN as u64);
}

// ---------------------------------------------------------------------------
// DIVU (unsigned)
// ---------------------------------------------------------------------------

#[test]
fn test_divu_basic() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 100);
    write_reg(&mut state, REG_C, 10);
    let inst = make_instruction(Rv64DivRemOpcode::DIVU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 10);
}

#[test]
fn test_divu_by_zero() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 0);
    let inst = make_instruction(Rv64DivRemOpcode::DIVU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), u64::MAX);
}

// ---------------------------------------------------------------------------
// REM (signed)
// ---------------------------------------------------------------------------

#[test]
fn test_rem_basic() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 43);
    write_reg(&mut state, REG_C, 7);
    let inst = make_instruction(Rv64DivRemOpcode::REM, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1);
}

#[test]
fn test_rem_by_zero() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 0);
    let inst = make_instruction(Rv64DivRemOpcode::REM, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 42);
}

#[test]
fn test_rem_overflow() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // i64::MIN % -1 = 0 (overflow case)
    write_reg(&mut state, REG_B, i64::MIN as u64);
    write_reg(&mut state, REG_C, u64::MAX); // -1
    let inst = make_instruction(Rv64DivRemOpcode::REM, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

// ---------------------------------------------------------------------------
// REMU (unsigned)
// ---------------------------------------------------------------------------

#[test]
fn test_remu_basic() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 103);
    write_reg(&mut state, REG_C, 10);
    let inst = make_instruction(Rv64DivRemOpcode::REMU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 3);
}

#[test]
fn test_remu_by_zero() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 0);
    let inst = make_instruction(Rv64DivRemOpcode::REMU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 42);
}

#[test]
fn test_divrem_advances_pc() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 10);
    write_reg(&mut state, REG_C, 3);
    let inst = make_instruction(Rv64DivRemOpcode::DIV, REG_A, REG_B, REG_C);

    let pc = execute(&executor, &mut state, &inst);
    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_div_negative_by_positive_truncates_toward_zero() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, (-7i64) as u64);
    write_reg(&mut state, REG_C, 3);
    let inst = make_instruction(Rv64DivRemOpcode::DIV, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), (-2i64) as u64);
}

#[test]
fn test_div_positive_by_negative_truncates_toward_zero() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 7);
    write_reg(&mut state, REG_C, (-3i64) as u64);
    let inst = make_instruction(Rv64DivRemOpcode::DIV, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), (-2i64) as u64);
}

#[test]
fn test_rem_signed_keeps_dividend_sign() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, (-7i64) as u64);
    write_reg(&mut state, REG_C, 3);
    let inst = make_instruction(Rv64DivRemOpcode::REM, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), (-1i64) as u64);
}

#[test]
fn test_rem_signed_negative_divisor() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 7);
    write_reg(&mut state, REG_C, (-3i64) as u64);
    let inst = make_instruction(Rv64DivRemOpcode::REM, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1);
}

#[test]
fn test_divu_large_values() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, u64::MAX);
    write_reg(&mut state, REG_C, 2);
    let inst = make_instruction(Rv64DivRemOpcode::DIVU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), u64::MAX / 2);
}

#[test]
fn test_div_zero_divided_by_nonzero() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0);
    write_reg(&mut state, REG_C, 42);
    let inst = make_instruction(Rv64DivRemOpcode::DIV, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_rem_exact_division() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 7);
    let inst = make_instruction(Rv64DivRemOpcode::REM, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_remu_large_values() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, u64::MAX);
    write_reg(&mut state, REG_C, 10);
    let inst = make_instruction(Rv64DivRemOpcode::REMU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), u64::MAX % 10);
}

#[test]
fn test_div_register_aliasing_rd_eq_rs1() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 42);
    write_reg(&mut state, REG_C, 7);
    let inst = make_instruction(Rv64DivRemOpcode::DIV, REG_A, REG_A, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 6);
}

#[test]
#[should_panic]
fn test_divrem_invalid_instruction_rejected() {
    let executor = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let inst = Instruction::from_usize::<7>(
        VmOpcode::from_usize(Rv64DivRemOpcode::DIV.global_opcode_usize()),
        [
            REG_A as usize,
            REG_B as usize,
            REG_C as usize,
            0, // invalid d
            RV32_REGISTER_AS as usize,
            0,
            0,
        ],
    );
    let _ = execute(&executor, &mut state, &inst);
}
