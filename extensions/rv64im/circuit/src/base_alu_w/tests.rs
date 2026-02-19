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
use openvm_rv64im_transpiler::Rv64BaseAluWOpcode;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use super::execution::sign_extend_32_to_64;
use crate::{
    test_utils::{create_exec_state, execute_instruction, read_reg, write_reg},
    Rv64BaseAluWExecutor,
};

type F = BabyBear;

const REG_A: u32 = 0;
const REG_B: u32 = 8;
const REG_C: u32 = 16;
const START_PC: u32 = 0x1000;

fn make_reg_instruction(opcode: Rv64BaseAluWOpcode, rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
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

fn make_imm_instruction(opcode: Rv64BaseAluWOpcode, rd: u32, rs1: u32, imm: u32) -> Instruction<F> {
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
    executor: &Rv64BaseAluWExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64BaseAluWOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}

// ---------------------------------------------------------------------------
// sign_extend_32_to_64 helper tests
// ---------------------------------------------------------------------------

#[test]
fn test_sign_extend_32_to_64_positive() {
    assert_eq!(sign_extend_32_to_64(0), 0u64);
    assert_eq!(sign_extend_32_to_64(1), 1u64);
    assert_eq!(sign_extend_32_to_64(0x7FFFFFFF), 0x7FFFFFFFu64);
}

#[test]
fn test_sign_extend_32_to_64_negative() {
    assert_eq!(sign_extend_32_to_64(0x80000000), 0xFFFFFFFF80000000u64);
    assert_eq!(sign_extend_32_to_64(0xFFFFFFFF), 0xFFFFFFFFFFFFFFFFu64);
}

// ---------------------------------------------------------------------------
// ADDW register tests (execute through InterpreterExecutor)
// ---------------------------------------------------------------------------

#[test]
fn test_addw_reg_basic() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 100);
    write_reg(&mut state, REG_C, 200);
    let inst = make_reg_instruction(Rv64BaseAluWOpcode::ADDW, REG_A, REG_B, REG_C);
    let new_pc = execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 300);
    assert_eq!(new_pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_addw_overflow_sign_extends() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 0x7FFFFFFF + 1 overflows to 0x80000000, sign-extended to 0xFFFFFFFF80000000
    write_reg(&mut state, REG_B, 0x7FFFFFFF);
    write_reg(&mut state, REG_C, 1);
    let inst = make_reg_instruction(Rv64BaseAluWOpcode::ADDW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xFFFFFFFF80000000);
}

#[test]
fn test_addw_truncates_upper_bits() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Upper 32 bits should be ignored for the addition
    write_reg(&mut state, REG_B, 0xDEAD_BEEF_0000_0001);
    write_reg(&mut state, REG_C, 0xCAFE_BABE_0000_0002);
    let inst = make_reg_instruction(Rv64BaseAluWOpcode::ADDW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // Only lower 32 bits matter: 1 + 2 = 3, positive -> zero-extended
    assert_eq!(read_reg(&state, REG_A), 3);
}

// ---------------------------------------------------------------------------
// SUBW register tests
// ---------------------------------------------------------------------------

#[test]
fn test_subw_reg_basic() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 500);
    write_reg(&mut state, REG_C, 200);
    let inst = make_reg_instruction(Rv64BaseAluWOpcode::SUBW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 300);
}

#[test]
fn test_subw_underflow_sign_extends() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 0 - 1 = 0xFFFFFFFF in 32-bit, sign-extended to 0xFFFFFFFFFFFFFFFF
    write_reg(&mut state, REG_B, 0);
    write_reg(&mut state, REG_C, 1);
    let inst = make_reg_instruction(Rv64BaseAluWOpcode::SUBW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xFFFFFFFFFFFFFFFF);
}

#[test]
fn test_subw_truncates_upper_bits() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Upper 32 bits should be ignored
    write_reg(&mut state, REG_B, 0xAAAAAAAA_00000005);
    write_reg(&mut state, REG_C, 0xBBBBBBBB_00000003);
    let inst = make_reg_instruction(Rv64BaseAluWOpcode::SUBW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 2);
}

// ---------------------------------------------------------------------------
// ADDIW (ADDW with immediate)
// ---------------------------------------------------------------------------

#[test]
fn test_addiw_positive_imm() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1000);
    let inst = make_imm_instruction(Rv64BaseAluWOpcode::ADDW, REG_A, REG_B, 42);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1042);
}

#[test]
fn test_addiw_negative_imm() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1000);
    // 0xFFFFFF is -1 in 24-bit signed
    let inst = make_imm_instruction(Rv64BaseAluWOpcode::ADDW, REG_A, REG_B, 0xFFFFFF);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 999);
}

#[test]
fn test_addiw_overflow_sign_extends() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x7FFFFFFF);
    let inst = make_imm_instruction(Rv64BaseAluWOpcode::ADDW, REG_A, REG_B, 1);
    execute(&executor, &mut state, &inst);

    // 0x7FFFFFFF + 1 = 0x80000000, sign-extended
    assert_eq!(read_reg(&state, REG_A), 0xFFFFFFFF80000000);
}

// ---------------------------------------------------------------------------
// PC advancement
// ---------------------------------------------------------------------------

#[test]
fn test_addw_advances_pc() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 2);
    let inst = make_reg_instruction(Rv64BaseAluWOpcode::ADDW, REG_A, REG_B, REG_C);

    let pc1 = execute(&executor, &mut state, &inst);
    assert_eq!(pc1, START_PC + DEFAULT_PC_STEP);

    let pc2 = execute(&executor, &mut state, &inst);
    assert_eq!(pc2, START_PC + 2 * DEFAULT_PC_STEP);
}

// ---------------------------------------------------------------------------
// Destination register isolation
// ---------------------------------------------------------------------------

#[test]
fn test_addw_does_not_clobber_sources() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 58);
    let inst = make_reg_instruction(Rv64BaseAluWOpcode::ADDW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 100);
    assert_eq!(read_reg(&state, REG_B), 42);
    assert_eq!(read_reg(&state, REG_C), 58);
}

#[test]
fn test_addw_result_zero() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 0xFFFFFFFF + 1 = 0x00000000 in 32 bits, zero-extended
    write_reg(&mut state, REG_B, 0xFFFFFFFF);
    write_reg(&mut state, REG_C, 1);
    let inst = make_reg_instruction(Rv64BaseAluWOpcode::ADDW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_subw_negative_result() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 10 - 20 = -10 in 32-bit, sign-extended to 64 bits
    write_reg(&mut state, REG_B, 10);
    write_reg(&mut state, REG_C, 20);
    let inst = make_reg_instruction(Rv64BaseAluWOpcode::SUBW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), (-10i32) as i64 as u64);
}

#[test]
fn test_addw_register_aliasing_rd_eq_rs1() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 100);
    write_reg(&mut state, REG_C, 200);
    let inst = make_reg_instruction(Rv64BaseAluWOpcode::ADDW, REG_A, REG_A, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 300);
}

#[test]
fn test_subw_i32_min() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // i32::MIN - 1 overflows: 0x80000000 - 1 = 0x7FFFFFFF
    write_reg(&mut state, REG_B, 0x80000000);
    write_reg(&mut state, REG_C, 1);
    let inst = make_reg_instruction(Rv64BaseAluWOpcode::SUBW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0x7FFFFFFF);
}

#[test]
#[should_panic]
fn test_subw_immediate_invalid_instruction_rejected() {
    let executor = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let inst = make_imm_instruction(Rv64BaseAluWOpcode::SUBW, REG_A, REG_B, 1);
    let _ = execute(&executor, &mut state, &inst);
}
