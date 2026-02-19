use openvm_circuit::{
    arch::{execution_mode::ExecutionCtx, VmExecState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
    LocalOpcode, VmOpcode,
};
use openvm_rv64im_transpiler::Rv64ShiftWOpcode;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use crate::{
    test_utils::{create_exec_state, execute_instruction, read_reg, write_reg},
    Rv64ShiftWExecutor,
};

type F = BabyBear;

const REG_A: u32 = 0;
const REG_B: u32 = 8;
const REG_C: u32 = 16;
const START_PC: u32 = 0x1000;

fn make_reg_instruction(opcode: Rv64ShiftWOpcode, rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
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

fn make_imm_instruction(opcode: Rv64ShiftWOpcode, rd: u32, rs1: u32, imm: u32) -> Instruction<F> {
    Instruction::from_usize::<7>(
        VmOpcode::from_usize(opcode.global_opcode_usize()),
        [
            rd as usize,
            rs1 as usize,
            imm as usize,
            RV32_REGISTER_AS as usize,
            RV32_IMM_AS as usize,
            0,
            0,
        ],
    )
}

fn execute(
    executor: &Rv64ShiftWExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64ShiftWOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}

// ---------------------------------------------------------------------------
// SLLW
// ---------------------------------------------------------------------------

#[test]
fn test_sllw_basic() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 10);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SLLW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1 << 10);
}

#[test]
fn test_sllw_sign_extends_result() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 1 << 31 = 0x80000000, sign-extended to 0xFFFFFFFF80000000
    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 31);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SLLW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xFFFFFFFF80000000);
}

#[test]
fn test_sllw_truncates_upper_bits() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Upper 32 bits of rs1 should be ignored
    write_reg(&mut state, REG_B, 0xDEADBEEF_00000001);
    write_reg(&mut state, REG_C, 4);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SLLW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 16);
}

#[test]
fn test_sllw_5bit_mask() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // W-variants mask shift to 5 bits (0x1F), so 32 & 0x1F = 0
    write_reg(&mut state, REG_B, 0xFF);
    write_reg(&mut state, REG_C, 32);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SLLW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xFF);
}

// ---------------------------------------------------------------------------
// SRLW
// ---------------------------------------------------------------------------

#[test]
fn test_srlw_basic() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x80000000);
    write_reg(&mut state, REG_C, 31);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SRLW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1);
}

#[test]
fn test_srlw_fills_zeros() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0xFFFFFFFF);
    write_reg(&mut state, REG_C, 16);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SRLW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // 0xFFFFFFFF >> 16 = 0x0000FFFF, positive so zero-extended
    assert_eq!(read_reg(&state, REG_A), 0x0000FFFF);
}

// ---------------------------------------------------------------------------
// SRAW
// ---------------------------------------------------------------------------

#[test]
fn test_sraw_positive() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x7FFFFFFF);
    write_reg(&mut state, REG_C, 16);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SRAW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0x00007FFF);
}

#[test]
fn test_sraw_negative_fills_ones_and_sign_extends() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 0x80000000 as i32 = -2147483648
    write_reg(&mut state, REG_B, 0x80000000);
    write_reg(&mut state, REG_C, 1);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SRAW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // 0x80000000 >> 1 = 0xC0000000 (arithmetic), sign-extended to 64 bits
    assert_eq!(read_reg(&state, REG_A), 0xFFFFFFFFC0000000);
}

// ---------------------------------------------------------------------------
// Immediate tests
// ---------------------------------------------------------------------------

#[test]
fn test_sllw_imm() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    let inst = make_imm_instruction(Rv64ShiftWOpcode::SLLW, REG_A, REG_B, 5);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 32);
}

// ---------------------------------------------------------------------------
// PC advancement
// ---------------------------------------------------------------------------

#[test]
fn test_shift_w_advances_pc() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 1);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SLLW, REG_A, REG_B, REG_C);

    let pc1 = execute(&executor, &mut state, &inst);
    assert_eq!(pc1, START_PC + DEFAULT_PC_STEP);
}

// ---------------------------------------------------------------------------
// Missing immediate mode tests for SRLW and SRAW
// ---------------------------------------------------------------------------

#[test]
fn test_srlw_imm() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x80000000);
    let inst = make_imm_instruction(Rv64ShiftWOpcode::SRLW, REG_A, REG_B, 16);
    execute(&executor, &mut state, &inst);

    // 0x80000000 >> 16 = 0x00008000, positive so zero-extended
    assert_eq!(read_reg(&state, REG_A), 0x00008000);
}

#[test]
fn test_sraw_imm_negative() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x80000000); // i32::MIN
    let inst = make_imm_instruction(Rv64ShiftWOpcode::SRAW, REG_A, REG_B, 4);
    execute(&executor, &mut state, &inst);

    // 0x80000000 >> 4 = 0xF8000000 (arithmetic), sign-extended to 64 bits
    assert_eq!(read_reg(&state, REG_A), 0xFFFFFFFFF8000000);
}

// ---------------------------------------------------------------------------
// SRLW/SRAW 5-bit mask wrapping
// ---------------------------------------------------------------------------

#[test]
fn test_srlw_5bit_mask() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 32 & 0x1F = 0, so shift amount is 0 (identity in 32 bits)
    write_reg(&mut state, REG_B, 0xFF);
    write_reg(&mut state, REG_C, 32);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SRLW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xFF);
}

#[test]
fn test_sraw_5bit_mask() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 32 & 0x1F = 0, identity
    write_reg(&mut state, REG_B, 0x80000000); // negative in 32-bit
    write_reg(&mut state, REG_C, 32);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SRAW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // Identity → 0x80000000 sign-extended to 0xFFFFFFFF80000000
    assert_eq!(read_reg(&state, REG_A), 0xFFFFFFFF80000000);
}

// ---------------------------------------------------------------------------
// Shift by zero (identity)
// ---------------------------------------------------------------------------

#[test]
fn test_sllw_by_zero() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x12345678);
    write_reg(&mut state, REG_C, 0);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SLLW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0x12345678);
}

#[test]
fn test_srlw_by_zero() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x80000000);
    write_reg(&mut state, REG_C, 0);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SRLW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // 0x80000000 with no shift, then sign-extended
    assert_eq!(read_reg(&state, REG_A), 0xFFFFFFFF80000000);
}

// ---------------------------------------------------------------------------
// SLLW shift out all bits
// ---------------------------------------------------------------------------

#[test]
fn test_sllw_shift_31_shifts_out() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 0xFFFFFFFF << 31 = 0x80000000 in 32 bits, sign-extended
    write_reg(&mut state, REG_B, 0xFFFFFFFF);
    write_reg(&mut state, REG_C, 31);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SLLW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xFFFFFFFF80000000);
}

// ---------------------------------------------------------------------------
// Register aliasing
// ---------------------------------------------------------------------------

#[test]
fn test_sllw_register_aliasing_rd_eq_rs1() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 1);
    write_reg(&mut state, REG_C, 4);
    let inst = make_reg_instruction(Rv64ShiftWOpcode::SLLW, REG_A, REG_A, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 16);
}

#[test]
#[should_panic]
fn test_shift_w_invalid_instruction_rejected() {
    let executor = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Invalid: e must be RV32_IMM_AS or RV32_REGISTER_AS.
    let inst = Instruction::from_usize::<7>(
        VmOpcode::from_usize(Rv64ShiftWOpcode::SLLW.global_opcode_usize()),
        [
            REG_A as usize,
            REG_B as usize,
            REG_C as usize,
            RV32_REGISTER_AS as usize,
            999,
            0,
            0,
        ],
    );
    execute(&executor, &mut state, &inst);
}
