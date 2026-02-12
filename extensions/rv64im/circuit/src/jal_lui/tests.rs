use openvm_circuit::{
    arch::{execution_mode::ExecutionCtx, VmExecState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode, VmOpcode,
};
use openvm_rv64im_transpiler::Rv64JalLuiOpcode;
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use crate::{
    test_utils::{create_exec_state, execute_instruction, read_reg},
    Rv64JalLuiExecutor,
};

type F = BabyBear;

const REG_A: u32 = 0;
const START_PC: u32 = 0x1000;

fn isize_to_field(value: isize) -> F {
    if value < 0 {
        return F::NEG_ONE * F::from_canonical_usize(value.unsigned_abs());
    }
    F::from_canonical_usize(value as usize)
}

fn make_jal_instruction(rd: u32, imm: i32, enabled: bool) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(Rv64JalLuiOpcode::JAL.global_opcode_usize()),
        F::from_canonical_u32(rd),
        F::ZERO,
        isize_to_field(imm as isize),
        F::ONE,
        F::ZERO,
        F::from_bool(enabled),
        F::ZERO,
    )
}

fn make_lui_instruction(rd: u32, imm20: u32) -> Instruction<F> {
    // imm20 is the 20-bit upper immediate (the value that gets shifted left by 12)
    Instruction::new(
        VmOpcode::from_usize(Rv64JalLuiOpcode::LUI.global_opcode_usize()),
        F::from_canonical_u32(rd),
        F::ZERO,
        F::from_canonical_u32(imm20),
        F::ONE,
        F::ZERO,
        F::ONE,
        F::ZERO,
    )
}

fn execute(
    executor: &Rv64JalLuiExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64JalLuiOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}

#[test]
fn test_jal_jumps_and_writes_rd() {
    let executor = Rv64JalLuiExecutor::new(Rv64JalLuiOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let inst = make_jal_instruction(REG_A, 200, true);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + 200);
    assert_eq!(
        read_reg(&mut state, REG_A),
        (START_PC + DEFAULT_PC_STEP) as u64
    );
}

#[test]
fn test_jal_negative_offset() {
    let executor = Rv64JalLuiExecutor::new(Rv64JalLuiOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let inst = make_jal_instruction(REG_A, -100, true);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC.wrapping_sub(100));
    assert_eq!(
        read_reg(&mut state, REG_A),
        (START_PC + DEFAULT_PC_STEP) as u64
    );
}

#[test]
fn test_jal_disabled_does_not_write_rd() {
    let executor = Rv64JalLuiExecutor::new(Rv64JalLuiOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let inst = make_jal_instruction(REG_A, 200, false);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + 200);
    // rd should not be written (remains 0)
    assert_eq!(read_reg(&mut state, REG_A), 0);
}

#[test]
fn test_lui_basic() {
    let executor = Rv64JalLuiExecutor::new(Rv64JalLuiOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // LUI with imm20 = 1 → result = 1 << 12 = 0x1000
    let inst = make_lui_instruction(REG_A, 1);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
    assert_eq!(read_reg(&mut state, REG_A), 0x1000);
}

#[test]
fn test_lui_sign_extends_to_64_bits() {
    let executor = Rv64JalLuiExecutor::new(Rv64JalLuiOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // LUI with imm20 = 0x80000 → result = 0x80000 << 12 = 0x80000000
    // This has bit 31 set, so sign-extension to 64 bits gives 0xFFFFFFFF_80000000
    let inst = make_lui_instruction(REG_A, 0x80000);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
    assert_eq!(read_reg(&mut state, REG_A), 0xFFFFFFFF_80000000u64);
}

#[test]
fn test_lui_large_positive() {
    let executor = Rv64JalLuiExecutor::new(Rv64JalLuiOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // LUI with imm20 = 0x7FFFF → result = 0x7FFFF << 12 = 0x7FFFF000
    // Bit 31 is 0, so sign extension keeps it positive
    let inst = make_lui_instruction(REG_A, 0x7FFFF);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
    assert_eq!(read_reg(&mut state, REG_A), 0x7FFFF000u64);
}

#[test]
fn test_lui_zero() {
    let executor = Rv64JalLuiExecutor::new(Rv64JalLuiOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let inst = make_lui_instruction(REG_A, 0);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
    assert_eq!(read_reg(&mut state, REG_A), 0);
}

#[test]
fn test_lui_max_imm20() {
    let executor = Rv64JalLuiExecutor::new(Rv64JalLuiOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // max imm20 = 0xFFFFF → result = 0xFFFFF << 12 = 0xFFFFF000
    // Bit 31 is set → sign-extends to 0xFFFFFFFF_FFFFF000
    let inst = make_lui_instruction(REG_A, 0xFFFFF);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
    assert_eq!(read_reg(&mut state, REG_A), 0xFFFFFFFF_FFFFF000u64);
}

// JAL with offset=0 jumps to its own address, creating a self-loop.
// In TCO mode the interpreter re-dispatches indefinitely, so skip this test.
#[cfg(not(feature = "tco"))]
#[test]
fn test_jal_zero_offset() {
    let executor = Rv64JalLuiExecutor::new(Rv64JalLuiOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let inst = make_jal_instruction(REG_A, 0, true);
    let pc = execute(&executor, &mut state, &inst);

    // Jump to current PC (loop)
    assert_eq!(pc, START_PC);
    assert_eq!(
        read_reg(&mut state, REG_A),
        (START_PC + DEFAULT_PC_STEP) as u64
    );
}

#[test]
fn test_lui_one() {
    let executor = Rv64JalLuiExecutor::new(Rv64JalLuiOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // imm20 = 1 → result = 0x1000
    let inst = make_lui_instruction(REG_A, 1);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&mut state, REG_A), 0x1000u64);
}

#[test]
#[should_panic]
fn test_jal_lui_invalid_instruction_rejected() {
    let executor = Rv64JalLuiExecutor::new(Rv64JalLuiOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Invalid: d must be RV32_REGISTER_AS.
    let inst = Instruction::new(
        VmOpcode::from_usize(Rv64JalLuiOpcode::JAL.global_opcode_usize()),
        F::from_canonical_u32(REG_A),
        F::ZERO,
        F::from_canonical_u32(4),
        F::ZERO,
        F::ZERO,
        F::ONE,
        F::ZERO,
    );
    execute(&executor, &mut state, &inst);
}
