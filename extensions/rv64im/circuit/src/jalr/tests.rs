use crate::test_utils::{create_exec_state, execute_instruction, read_reg, write_reg};
use openvm_circuit::{
    arch::{execution_mode::ExecutionCtx, VmExecState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode,
    VmOpcode,
};
use openvm_rv64im_transpiler::Rv64JalrOpcode;
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use crate::Rv64JalrExecutor;

type F = BabyBear;

const REG_A: u32 = 0;
const REG_B: u32 = 8;
const START_PC: u32 = 0x1000;

fn make_jalr_instruction(rd: u32, rs1: u32, imm: i32, enabled: bool) -> Instruction<F> {
    // imm_extended = c + g * 0xffff0000
    // c = (imm as u32) & 0xffff
    // g = 1 if imm < 0, 0 otherwise
    let c = (imm as u32) & 0xffff;
    let g = if imm < 0 { 1u32 } else { 0u32 };
    Instruction::new(
        VmOpcode::from_usize(Rv64JalrOpcode::JALR.global_opcode_usize()),
        F::from_canonical_u32(rd),
        F::from_canonical_u32(rs1),
        F::from_canonical_u32(c),
        F::ONE,
        F::ZERO,
        F::from_bool(enabled),
        F::from_canonical_u32(g),
    )
}

fn execute(
    executor: &Rv64JalrExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64JalrOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}

#[test]
fn test_jalr_basic() {
    let executor = Rv64JalrExecutor::new(Rv64JalrOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x2000);
    let inst = make_jalr_instruction(REG_A, REG_B, 100, true);
    let pc = execute(&executor, &mut state, &inst);

    // to_pc = (0x2000 + 100) & ~1 = 0x2064
    assert_eq!(pc, 0x2064);
    assert_eq!(
        read_reg(&mut state, REG_A),
        (START_PC + DEFAULT_PC_STEP) as u64
    );
}

#[test]
fn test_jalr_negative_offset() {
    let executor = Rv64JalrExecutor::new(Rv64JalrOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x3000);
    let inst = make_jalr_instruction(REG_A, REG_B, -100, true);
    let pc = execute(&executor, &mut state, &inst);

    // to_pc = (0x3000 + (-100)) & ~1 = (0x3000 - 100) & ~1 = 0x2F9C
    assert_eq!(pc, 0x2F9C);
    assert_eq!(
        read_reg(&mut state, REG_A),
        (START_PC + DEFAULT_PC_STEP) as u64
    );
}

#[test]
fn test_jalr_clears_bit_0() {
    let executor = Rv64JalrExecutor::new(Rv64JalrOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x2001);
    let inst = make_jalr_instruction(REG_A, REG_B, 0, true);
    let pc = execute(&executor, &mut state, &inst);

    // to_pc = (0x2001 + 0) & ~1 = 0x2000
    assert_eq!(pc, 0x2000);
}

#[test]
fn test_jalr_disabled_does_not_write_rd() {
    let executor = Rv64JalrExecutor::new(Rv64JalrOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x2000);
    let inst = make_jalr_instruction(REG_A, REG_B, 0, false);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, 0x2000);
    // rd should not be written (remains 0)
    assert_eq!(read_reg(&mut state, REG_A), 0);
}

#[test]
fn test_jalr_small_rs1_value() {
    let executor = Rv64JalrExecutor::new(Rv64JalrOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // In practice, the PC is 32-bit, so after computation the result is truncated to u32.
    // But rs1 is read as 64 bits. The wrapping_add with sign-extended imm then truncation to u32.
    write_reg(&mut state, REG_B, 0x100);
    let inst = make_jalr_instruction(REG_A, REG_B, 8, true);
    let pc = execute(&executor, &mut state, &inst);

    // to_pc = (0x100 + 8) & ~1 = 0x108
    assert_eq!(pc, 0x108);
}

#[test]
fn test_jalr_large_64bit_rs1() {
    let executor = Rv64JalrExecutor::new(Rv64JalrOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // rs1 has bits above 32. After wrapping_add + truncation to u32, result should be correct.
    write_reg(&mut state, REG_B, 0x1_0000_2000u64);
    let inst = make_jalr_instruction(REG_A, REG_B, 4, true);
    let pc = execute(&executor, &mut state, &inst);

    // (0x1_0000_2000 + 4) & ~1 truncated to u32 = 0x00002004
    assert_eq!(pc, 0x2004);
}

#[test]
fn test_jalr_imm_minus_one() {
    let executor = Rv64JalrExecutor::new(Rv64JalrOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x2002);
    let inst = make_jalr_instruction(REG_A, REG_B, -1, true);
    let pc = execute(&executor, &mut state, &inst);

    // (0x2002 + (-1)) & ~1 = 0x2001 & ~1 = 0x2000
    assert_eq!(pc, 0x2000);
    assert_eq!(
        read_reg(&mut state, REG_A),
        (START_PC + DEFAULT_PC_STEP) as u64
    );
}

#[test]
#[should_panic]
fn test_jalr_invalid_instruction_rejected() {
    let executor = Rv64JalrExecutor::new(Rv64JalrOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Invalid: d must be RV32_REGISTER_AS.
    let inst = Instruction::new(
        VmOpcode::from_usize(Rv64JalrOpcode::JALR.global_opcode_usize()),
        F::from_canonical_u32(REG_A),
        F::from_canonical_u32(REG_B),
        F::from_canonical_u32(0),
        F::ZERO,
        F::ZERO,
        F::ONE,
        F::ZERO,
    );
    execute(&executor, &mut state, &inst);
}
