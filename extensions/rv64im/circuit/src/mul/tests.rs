use crate::test_utils::{create_exec_state, execute_instruction, read_reg, write_reg};
use openvm_circuit::{
    arch::{execution_mode::ExecutionCtx, VmExecState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
    VmOpcode,
};
use openvm_rv64im_transpiler::Rv64MulOpcode;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use crate::Rv64MulExecutor;

type F = BabyBear;

const REG_A: u32 = 0;
const REG_B: u32 = 8;
const REG_C: u32 = 16;
const START_PC: u32 = 0x1000;

fn make_instruction(rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
    Instruction::from_usize::<7>(
        VmOpcode::from_usize(Rv64MulOpcode::MUL.global_opcode_usize()),
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
    executor: &Rv64MulExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64MulOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}

#[test]
fn test_mul_basic() {
    let executor = Rv64MulExecutor::new(Rv64MulOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 6);
    write_reg(&mut state, REG_C, 7);
    let inst = make_instruction(REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 42);
}

#[test]
fn test_mul_by_zero() {
    let executor = Rv64MulExecutor::new(Rv64MulOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 12345);
    write_reg(&mut state, REG_C, 0);
    let inst = make_instruction(REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_mul_wrapping_overflow() {
    let executor = Rv64MulExecutor::new(Rv64MulOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, u64::MAX);
    write_reg(&mut state, REG_C, 2);
    let inst = make_instruction(REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), u64::MAX.wrapping_mul(2));
}

#[test]
fn test_mul_large_values() {
    let executor = Rv64MulExecutor::new(Rv64MulOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0x1_0000_0000);
    write_reg(&mut state, REG_C, 0x1_0000_0000);
    let inst = make_instruction(REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // Lower 64 bits of (2^32 * 2^32) = 2^64, which wraps to 0
    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_mul_advances_pc() {
    let executor = Rv64MulExecutor::new(Rv64MulOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 1);
    let inst = make_instruction(REG_A, REG_B, REG_C);

    let pc = execute(&executor, &mut state, &inst);
    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_mul_by_one() {
    let executor = Rv64MulExecutor::new(Rv64MulOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0xDEADBEEFCAFEBABE);
    write_reg(&mut state, REG_C, 1);
    let inst = make_instruction(REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xDEADBEEFCAFEBABE);
}

#[test]
fn test_mul_negative_times_negative() {
    let executor = Rv64MulExecutor::new(Rv64MulOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // (-2) * (-3) = 6 (lower 64 bits)
    write_reg(&mut state, REG_B, (-2i64) as u64);
    write_reg(&mut state, REG_C, (-3i64) as u64);
    let inst = make_instruction(REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 6);
}

#[test]
#[should_panic]
fn test_mul_invalid_instruction_rejected() {
    let executor = Rv64MulExecutor::new(Rv64MulOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Invalid: d must be RV32_REGISTER_AS.
    let inst = Instruction::from_usize::<7>(
        VmOpcode::from_usize(Rv64MulOpcode::MUL.global_opcode_usize()),
        [
            REG_A as usize,
            REG_B as usize,
            REG_C as usize,
            0,
            RV32_REGISTER_AS as usize,
            0,
            0,
        ],
    );
    execute(&executor, &mut state, &inst);
}
