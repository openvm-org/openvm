use crate::test_utils::{create_exec_state, execute_instruction, read_reg, write_reg};
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
use openvm_rv64im_transpiler::Rv64LessThanOpcode;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use crate::Rv64LessThanExecutor;

type F = BabyBear;

const REG_A: u32 = 0;
const REG_B: u32 = 8;
const REG_C: u32 = 16;
const START_PC: u32 = 0x1000;

fn make_reg_instruction(opcode: Rv64LessThanOpcode, rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
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

fn make_imm_instruction(opcode: Rv64LessThanOpcode, rd: u32, rs1: u32, imm: u32) -> Instruction<F> {
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
    executor: &Rv64LessThanExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64LessThanOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}

// ---------------------------------------------------------------------------
// SLT (signed) register tests
// ---------------------------------------------------------------------------

#[test]
fn test_slt_less() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 10);
    write_reg(&mut state, REG_C, 20);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLT, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1);
}

#[test]
fn test_slt_equal() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 42);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLT, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_slt_greater() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 100);
    write_reg(&mut state, REG_C, 50);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLT, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_slt_negative_less_than_positive() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // -1 (as i64) < 1
    write_reg(&mut state, REG_B, u64::MAX); // -1
    write_reg(&mut state, REG_C, 1);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLT, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1);
}

#[test]
fn test_slt_i64_min_less_than_i64_max() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, i64::MIN as u64);
    write_reg(&mut state, REG_C, i64::MAX as u64);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLT, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1);
}

// ---------------------------------------------------------------------------
// SLTU (unsigned) register tests
// ---------------------------------------------------------------------------

#[test]
fn test_sltu_less() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 10);
    write_reg(&mut state, REG_C, 20);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLTU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1);
}

#[test]
fn test_sltu_greater() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 20);
    write_reg(&mut state, REG_C, 10);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLTU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_sltu_max_not_less_than_zero() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // u64::MAX > 0 (unsigned)
    write_reg(&mut state, REG_B, u64::MAX);
    write_reg(&mut state, REG_C, 0);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLTU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_sltu_zero_less_than_max() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0);
    write_reg(&mut state, REG_C, u64::MAX);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLTU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1);
}

// ---------------------------------------------------------------------------
// Signed vs unsigned interpretation
// ---------------------------------------------------------------------------

#[test]
fn test_slt_vs_sltu_negative_value() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);

    // -1 (0xFFFFFFFFFFFFFFFF) vs 1
    // SLT: -1 < 1 → true
    let mut state = create_exec_state(START_PC);
    write_reg(&mut state, REG_B, u64::MAX);
    write_reg(&mut state, REG_C, 1);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLT, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);
    assert_eq!(read_reg(&state, REG_A), 1);

    // SLTU: 0xFFFF...FFFF > 1 → false
    let mut state = create_exec_state(START_PC);
    write_reg(&mut state, REG_B, u64::MAX);
    write_reg(&mut state, REG_C, 1);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLTU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);
    assert_eq!(read_reg(&state, REG_A), 0);
}

// ---------------------------------------------------------------------------
// Immediate tests
// ---------------------------------------------------------------------------

#[test]
fn test_slt_imm_positive() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 5);
    let inst = make_imm_instruction(Rv64LessThanOpcode::SLT, REG_A, REG_B, 10);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1);
}

#[test]
fn test_sltu_imm() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 100);
    let inst = make_imm_instruction(Rv64LessThanOpcode::SLTU, REG_A, REG_B, 50);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_slt_imm_negative() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0);
    // 0xFFFFFF encodes -1 in 24-bit signed
    let inst = make_imm_instruction(Rv64LessThanOpcode::SLT, REG_A, REG_B, 0xFFFFFF);
    execute(&executor, &mut state, &inst);

    // 0 < -1 is false (signed)
    assert_eq!(read_reg(&state, REG_A), 0);
}

// ---------------------------------------------------------------------------
// PC advancement
// ---------------------------------------------------------------------------

#[test]
fn test_slt_advances_pc() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 2);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLT, REG_A, REG_B, REG_C);

    let pc1 = execute(&executor, &mut state, &inst);
    assert_eq!(pc1, START_PC + DEFAULT_PC_STEP);

    let pc2 = execute(&executor, &mut state, &inst);
    assert_eq!(pc2, START_PC + 2 * DEFAULT_PC_STEP);
}

// ---------------------------------------------------------------------------
// Result is always 0 or 1 (upper bytes are zero)
// ---------------------------------------------------------------------------

#[test]
fn test_result_only_lsb() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Pre-fill rd with garbage
    write_reg(&mut state, REG_A, 0xDEADBEEFCAFEBABE);
    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 2);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLT, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // Result should be exactly 1, not 0xDEADBEEFCAFEBA01 or similar
    assert_eq!(read_reg(&state, REG_A), 1);
}

// ---------------------------------------------------------------------------
// Missing edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_slt_negative_less_than_negative() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // -100 < -50 (signed)
    write_reg(&mut state, REG_B, (-100i64) as u64);
    write_reg(&mut state, REG_C, (-50i64) as u64);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLT, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1);
}

#[test]
#[should_panic]
fn test_less_than_invalid_instruction_rejected() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Invalid: e must be RV32_IMM_AS or RV32_REGISTER_AS.
    let inst = Instruction::from_usize::<7>(
        VmOpcode::from_usize(Rv64LessThanOpcode::SLT.global_opcode_usize()),
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

#[test]
fn test_slt_negative_not_less_than_more_negative() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // -50 < -100 is false
    write_reg(&mut state, REG_B, (-50i64) as u64);
    write_reg(&mut state, REG_C, (-100i64) as u64);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLT, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_sltu_equal_values() {
    let executor = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 42);
    let inst = make_reg_instruction(Rv64LessThanOpcode::SLTU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}
