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
use openvm_rv64im_transpiler::Rv64BaseAluOpcode;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use super::execution::imm24_sign_extend_to_u64;
use crate::Rv64BaseAluExecutor;

type F = BabyBear;

// Register byte offsets (RV64: 8 bytes per register)
const REG_A: u32 = 0; // rd
const REG_B: u32 = 8; // rs1
const REG_C: u32 = 16; // rs2

const START_PC: u32 = 0x1000;

fn make_reg_instruction(opcode: Rv64BaseAluOpcode, rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
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

fn make_imm_instruction(opcode: Rv64BaseAluOpcode, rd: u32, rs1: u32, imm: u32) -> Instruction<F> {
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

/// Execute an instruction via the InterpreterExecutor (E1) path and return the new PC.
fn execute(
    executor: &Rv64BaseAluExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64BaseAluOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}

// ---------------------------------------------------------------------------
// imm24_sign_extend_to_u64 helper tests
// ---------------------------------------------------------------------------

#[test]
fn test_imm24_sign_extend_positive() {
    assert_eq!(imm24_sign_extend_to_u64(1), 1u64);
    assert_eq!(imm24_sign_extend_to_u64(0x7FFFFF), 0x7FFFFFu64);
}

#[test]
fn test_imm24_sign_extend_negative() {
    assert_eq!(imm24_sign_extend_to_u64(0x800000), 0xFFFFFFFFFF800000u64);
    assert_eq!(imm24_sign_extend_to_u64(0xFFFFFF), 0xFFFFFFFFFFFFFFFFu64);
}

#[test]
fn test_imm24_sign_extend_zero() {
    assert_eq!(imm24_sign_extend_to_u64(0), 0u64);
}

// ---------------------------------------------------------------------------
// Register-register ALU tests (execute through InterpreterExecutor)
// ---------------------------------------------------------------------------

#[test]
fn test_add_reg() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 100);
    write_reg(&mut state, REG_C, 200);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::ADD, REG_A, REG_B, REG_C);
    let new_pc = execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 300);
    assert_eq!(new_pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_add_wrapping_overflow() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, u64::MAX);
    write_reg(&mut state, REG_C, 1);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::ADD, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_sub_reg() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 500);
    write_reg(&mut state, REG_C, 200);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::SUB, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 300);
}

#[test]
fn test_sub_wrapping_underflow() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0);
    write_reg(&mut state, REG_C, 1);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::SUB, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), u64::MAX);
}

#[test]
fn test_xor_reg() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0xFF00FF00_FF00FF00);
    write_reg(&mut state, REG_C, 0x0FF00FF0_0FF00FF0);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::XOR, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xF0F0F0F0_F0F0F0F0);
}

#[test]
fn test_or_reg() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0xFF00_0000_0000_0000);
    write_reg(&mut state, REG_C, 0x0000_0000_0000_00FF);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::OR, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xFF00_0000_0000_00FF);
}

#[test]
fn test_and_reg() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0xFF00FF00_FF00FF00);
    write_reg(&mut state, REG_C, 0x0FF00FF0_0FF00FF0);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::AND, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0x0F000F00_0F000F00);
}

// ---------------------------------------------------------------------------
// Immediate ALU tests
// ---------------------------------------------------------------------------

#[test]
fn test_add_imm_positive() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1000);
    // imm = 42 (positive 24-bit)
    let inst = make_imm_instruction(Rv64BaseAluOpcode::ADD, REG_A, REG_B, 42);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1042);
}

#[test]
fn test_add_imm_negative() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1000);
    // imm = 0xFFFFFF encodes -1 in 24-bit signed
    let inst = make_imm_instruction(Rv64BaseAluOpcode::ADD, REG_A, REG_B, 0xFFFFFF);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 999);
}

#[test]
fn test_and_imm() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0xFFFF_FFFF_FFFF_FFFF);
    // imm = 0xFF (positive)
    let inst = make_imm_instruction(Rv64BaseAluOpcode::AND, REG_A, REG_B, 0xFF);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xFF);
}

#[test]
fn test_or_imm_sign_extended() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0);
    // imm = 0x800000 (negative in 24-bit: sign-extends to 0xFFFFFFFFFF800000)
    let inst = make_imm_instruction(Rv64BaseAluOpcode::OR, REG_A, REG_B, 0x800000);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xFFFF_FFFF_FF80_0000);
}

// ---------------------------------------------------------------------------
// PC advancement
// ---------------------------------------------------------------------------

#[test]
fn test_pc_advances_each_instruction() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 2);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::ADD, REG_A, REG_B, REG_C);

    let pc1 = execute(&executor, &mut state, &inst);
    assert_eq!(pc1, START_PC + DEFAULT_PC_STEP);

    // Execute again - PC should advance further
    let pc2 = execute(&executor, &mut state, &inst);
    assert_eq!(pc2, START_PC + 2 * DEFAULT_PC_STEP);
}

// ---------------------------------------------------------------------------
// Destination register isolation
// ---------------------------------------------------------------------------

#[test]
fn test_rd_does_not_clobber_sources() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 58);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::ADD, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 100);
    assert_eq!(read_reg(&state, REG_B), 42);
    assert_eq!(read_reg(&state, REG_C), 58);
}

// ---------------------------------------------------------------------------
// Signed arithmetic edge cases (interpreted as u64 wrapping)
// ---------------------------------------------------------------------------

#[test]
fn test_add_signed_boundary() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // i64::MAX + 1 = i64::MIN (as u64)
    write_reg(&mut state, REG_B, i64::MAX as u64);
    write_reg(&mut state, REG_C, 1);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::ADD, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), i64::MIN as u64);
}

// ---------------------------------------------------------------------------
// Missing immediate mode tests
// ---------------------------------------------------------------------------

#[test]
fn test_sub_imm() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1000);
    let inst = make_imm_instruction(Rv64BaseAluOpcode::SUB, REG_A, REG_B, 42);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 958);
}

#[test]
fn test_xor_imm() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0xFF00);
    let inst = make_imm_instruction(Rv64BaseAluOpcode::XOR, REG_A, REG_B, 0xFF);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xFFFF);
}

// ---------------------------------------------------------------------------
// Bitwise identity tests
// ---------------------------------------------------------------------------

#[test]
fn test_xor_with_zero_is_identity() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0xDEADBEEFCAFEBABE);
    write_reg(&mut state, REG_C, 0);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::XOR, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xDEADBEEFCAFEBABE);
}

#[test]
fn test_or_with_zero_is_identity() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0xDEADBEEFCAFEBABE);
    write_reg(&mut state, REG_C, 0);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::OR, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xDEADBEEFCAFEBABE);
}

#[test]
fn test_and_with_all_ones_is_identity() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0xDEADBEEFCAFEBABE);
    write_reg(&mut state, REG_C, u64::MAX);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::AND, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xDEADBEEFCAFEBABE);
}

// ---------------------------------------------------------------------------
// Register aliasing (rd == rs1)
// ---------------------------------------------------------------------------

#[test]
fn test_add_register_aliasing_rd_eq_rs1() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 100);
    write_reg(&mut state, REG_C, 200);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::ADD, REG_A, REG_A, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 300);
}

#[test]
fn test_sub_register_aliasing_rd_eq_rs1() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 500);
    write_reg(&mut state, REG_C, 200);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::SUB, REG_A, REG_A, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 300);
}

#[test]
fn test_add_register_aliasing_rd_eq_rs2() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 100);
    write_reg(&mut state, REG_A, 200);
    let inst = make_reg_instruction(Rv64BaseAluOpcode::ADD, REG_A, REG_B, REG_A);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 300);
}

// ---------------------------------------------------------------------------
// AND/XOR immediate with sign-extended value
// ---------------------------------------------------------------------------

#[test]
fn test_and_imm_sign_extended() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0xFFFF_FFFF_FFFF_FFFF);
    // 0x800000 sign-extends to 0xFFFFFFFFFF800000
    let inst = make_imm_instruction(Rv64BaseAluOpcode::AND, REG_A, REG_B, 0x800000);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xFFFF_FFFF_FF80_0000);
}

#[test]
fn test_xor_imm_sign_extended() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0);
    // 0x800000 sign-extends to 0xFFFFFFFFFF800000
    let inst = make_imm_instruction(Rv64BaseAluOpcode::XOR, REG_A, REG_B, 0x800000);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xFFFF_FFFF_FF80_0000);
}

#[test]
#[should_panic]
fn test_base_alu_invalid_instruction_rejected() {
    let executor = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let inst = Instruction::from_usize::<7>(
        VmOpcode::from_usize(Rv64BaseAluOpcode::ADD.global_opcode_usize()),
        [
            REG_A as usize,
            REG_B as usize,
            REG_C as usize,
            0, // invalid d (must be RV32_REGISTER_AS)
            RV32_REGISTER_AS as usize,
            0,
            0,
        ],
    );

    let _ = execute(&executor, &mut state, &inst);
}
