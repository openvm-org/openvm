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
use openvm_rv64im_transpiler::Rv64ShiftOpcode;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use crate::Rv64ShiftExecutor;

type F = BabyBear;

const REG_A: u32 = 0;
const REG_B: u32 = 8;
const REG_C: u32 = 16;
const START_PC: u32 = 0x1000;

fn make_reg_instruction(opcode: Rv64ShiftOpcode, rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
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

fn make_imm_instruction(opcode: Rv64ShiftOpcode, rd: u32, rs1: u32, imm: u32) -> Instruction<F> {
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
    executor: &Rv64ShiftExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64ShiftOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}

// ---------------------------------------------------------------------------
// SLL (shift left logical)
// ---------------------------------------------------------------------------

#[test]
fn test_sll_basic() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 10);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SLL, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1 << 10);
}

#[test]
fn test_sll_high_bits() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 63);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SLL, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1u64 << 63);
}

#[test]
fn test_sll_6bit_mask() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // shift amount = 64 should be masked to 0 (64 & 0x3F = 0)
    write_reg(&mut state, REG_B, 0xFF);
    write_reg(&mut state, REG_C, 64);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SLL, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xFF);
}

// ---------------------------------------------------------------------------
// SRL (shift right logical)
// ---------------------------------------------------------------------------

#[test]
fn test_srl_basic() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x8000_0000_0000_0000);
    write_reg(&mut state, REG_C, 63);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SRL, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1);
}

#[test]
fn test_srl_fills_zeros() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // SRL fills with zeros from the left
    write_reg(&mut state, REG_B, u64::MAX);
    write_reg(&mut state, REG_C, 32);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SRL, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0x0000_0000_FFFF_FFFF);
}

// ---------------------------------------------------------------------------
// SRA (shift right arithmetic)
// ---------------------------------------------------------------------------

#[test]
fn test_sra_positive() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x7FFF_FFFF_FFFF_FFFF);
    write_reg(&mut state, REG_C, 32);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SRA, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // positive value: SRA fills with zeros, same as SRL
    assert_eq!(read_reg(&state, REG_A), 0x0000_0000_7FFF_FFFF);
}

#[test]
fn test_sra_negative_fills_ones() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // SRA on negative (MSB=1) fills with ones
    write_reg(&mut state, REG_B, u64::MAX); // -1 as i64
    write_reg(&mut state, REG_C, 32);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SRA, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), u64::MAX); // -1 >> anything = -1
}

#[test]
fn test_sra_sign_bit_preserved() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x8000_0000_0000_0000); // i64::MIN
    write_reg(&mut state, REG_C, 1);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SRA, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xC000_0000_0000_0000);
}

// ---------------------------------------------------------------------------
// Immediate tests
// ---------------------------------------------------------------------------

#[test]
fn test_sll_imm() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    let inst = make_imm_instruction(Rv64ShiftOpcode::SLL, REG_A, REG_B, 5);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 32);
}

#[test]
fn test_srl_imm() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 256);
    let inst = make_imm_instruction(Rv64ShiftOpcode::SRL, REG_A, REG_B, 4);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 16);
}

#[test]
fn test_sra_imm_negative() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x8000_0000_0000_0000); // i64::MIN
    let inst = make_imm_instruction(Rv64ShiftOpcode::SRA, REG_A, REG_B, 4);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xF800_0000_0000_0000);
}

// ---------------------------------------------------------------------------
// PC advancement
// ---------------------------------------------------------------------------

#[test]
fn test_shift_advances_pc() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 1);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SLL, REG_A, REG_B, REG_C);

    let pc1 = execute(&executor, &mut state, &inst);
    assert_eq!(pc1, START_PC + DEFAULT_PC_STEP);
}

// ---------------------------------------------------------------------------
// Shift by zero (identity)
// ---------------------------------------------------------------------------

#[test]
fn test_sll_by_zero() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0xDEADBEEFCAFEBABE);
    write_reg(&mut state, REG_C, 0);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SLL, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xDEADBEEFCAFEBABE);
}

#[test]
fn test_srl_by_zero() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0xDEADBEEFCAFEBABE);
    write_reg(&mut state, REG_C, 0);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SRL, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xDEADBEEFCAFEBABE);
}

#[test]
fn test_sra_by_zero() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x8000_0000_0000_0000);
    write_reg(&mut state, REG_C, 0);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SRA, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0x8000_0000_0000_0000);
}

// ---------------------------------------------------------------------------
// 32-bit boundary shift amounts (31, 32, 33)
// ---------------------------------------------------------------------------

#[test]
fn test_sll_by_31() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 31);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SLL, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1u64 << 31);
}

#[test]
fn test_sll_by_32() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 32);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SLL, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1u64 << 32);
}

#[test]
fn test_sll_by_33() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 33);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SLL, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1u64 << 33);
}

#[test]
fn test_srl_by_33() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, u64::MAX);
    write_reg(&mut state, REG_C, 33);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SRL, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), u64::MAX >> 33);
}

#[test]
fn test_sra_by_33() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x8000_0000_0000_0000); // i64::MIN
    write_reg(&mut state, REG_C, 33);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SRA, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), ((i64::MIN) >> 33) as u64);
}

// ---------------------------------------------------------------------------
// SRL/SRA 6-bit mask tests
// ---------------------------------------------------------------------------

#[test]
fn test_srl_6bit_mask() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 64 & 0x3F = 0, so shift amount is 0 (identity)
    write_reg(&mut state, REG_B, 0xDEADBEEFCAFEBABE);
    write_reg(&mut state, REG_C, 64);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SRL, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xDEADBEEFCAFEBABE);
}

#[test]
fn test_sra_6bit_mask() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 128 & 0x3F = 0, so shift amount is 0 (identity)
    write_reg(&mut state, REG_B, 0x8000_0000_0000_0000);
    write_reg(&mut state, REG_C, 128);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SRA, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0x8000_0000_0000_0000);
}

// ---------------------------------------------------------------------------
// Shift by 63 for SRL/SRA
// ---------------------------------------------------------------------------

#[test]
fn test_srl_by_63() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, u64::MAX);
    write_reg(&mut state, REG_C, 63);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SRL, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1);
}

#[test]
fn test_sra_by_63_negative() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x8000_0000_0000_0000); // i64::MIN
    write_reg(&mut state, REG_C, 63);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SRA, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // i64::MIN >> 63 = -1 (fills with ones)
    assert_eq!(read_reg(&state, REG_A), u64::MAX);
}

#[test]
fn test_sra_by_63_positive() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, i64::MAX as u64);
    write_reg(&mut state, REG_C, 63);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SRA, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

// ---------------------------------------------------------------------------
// Register aliasing
// ---------------------------------------------------------------------------

#[test]
fn test_sll_register_aliasing_rd_eq_rs1() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 1);
    write_reg(&mut state, REG_C, 10);
    let inst = make_reg_instruction(Rv64ShiftOpcode::SLL, REG_A, REG_A, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1 << 10);
}

#[test]
#[should_panic]
fn test_shift_invalid_instruction_rejected() {
    let executor = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Invalid: e must be RV32_IMM_AS or RV32_REGISTER_AS.
    let inst = Instruction::from_usize::<7>(
        VmOpcode::from_usize(Rv64ShiftOpcode::SLL.global_opcode_usize()),
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
