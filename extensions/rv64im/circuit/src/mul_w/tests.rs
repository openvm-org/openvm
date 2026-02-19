use openvm_circuit::{
    arch::{execution_mode::ExecutionCtx, VmExecState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
    VmOpcode,
};
use openvm_rv64im_transpiler::Rv64MulWOpcode;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use crate::{
    test_utils::{create_exec_state, execute_instruction, read_reg, write_reg},
    Rv64MulWExecutor,
};

type F = BabyBear;

const REG_A: u32 = 0;
const REG_B: u32 = 8;
const REG_C: u32 = 16;
const START_PC: u32 = 0x1000;

fn make_instruction(rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
    Instruction::from_usize::<7>(
        VmOpcode::from_usize(Rv64MulWOpcode::MULW.global_opcode_usize()),
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
    executor: &Rv64MulWExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64MulWOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}

#[test]
fn test_mulw_basic() {
    let executor = Rv64MulWExecutor::new(Rv64MulWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 6);
    write_reg(&mut state, REG_C, 7);
    let inst = make_instruction(REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 42);
}

#[test]
fn test_mulw_sign_extends_result() {
    let executor = Rv64MulWExecutor::new(Rv64MulWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 0x40000000 * 2 = 0x80000000, sign-extended to 0xFFFFFFFF80000000
    write_reg(&mut state, REG_B, 0x40000000);
    write_reg(&mut state, REG_C, 2);
    let inst = make_instruction(REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0xFFFFFFFF80000000);
}

#[test]
fn test_mulw_truncates_upper_bits() {
    let executor = Rv64MulWExecutor::new(Rv64MulWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 0xDEADBEEF_00000003);
    write_reg(&mut state, REG_C, 0xCAFEBABE_00000004);
    let inst = make_instruction(REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 12);
}

#[test]
fn test_mulw_advances_pc() {
    let executor = Rv64MulWExecutor::new(Rv64MulWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 1);
    let inst = make_instruction(REG_A, REG_B, REG_C);

    let pc = execute(&executor, &mut state, &inst);
    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_mulw_by_zero() {
    let executor = Rv64MulWExecutor::new(Rv64MulWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 12345);
    write_reg(&mut state, REG_C, 0);
    let inst = make_instruction(REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_mulw_by_one() {
    let executor = Rv64MulWExecutor::new(Rv64MulWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 1);
    let inst = make_instruction(REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 42);
}

#[test]
fn test_mulw_negative_times_negative() {
    let executor = Rv64MulWExecutor::new(Rv64MulWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // (-2i32) * (-3i32) = 6i32, sign-extended to 64 bits = 6
    write_reg(&mut state, REG_B, (-2i32) as u32 as u64);
    write_reg(&mut state, REG_C, (-3i32) as u32 as u64);
    let inst = make_instruction(REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 6);
}

#[test]
fn test_mulw_negative_times_positive() {
    let executor = Rv64MulWExecutor::new(Rv64MulWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // (-3i32) * 5i32 = -15i32, sign-extended
    write_reg(&mut state, REG_B, (-3i32) as u32 as u64);
    write_reg(&mut state, REG_C, 5);
    let inst = make_instruction(REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), (-15i32) as i64 as u64);
}

#[test]
fn test_mulw_wrapping_overflow() {
    let executor = Rv64MulWExecutor::new(Rv64MulWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 0x7FFFFFFF * 2 = 0xFFFFFFFE (wraps in 32 bits), sign-extended
    write_reg(&mut state, REG_B, 0x7FFFFFFF);
    write_reg(&mut state, REG_C, 2);
    let inst = make_instruction(REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    let result = (0x7FFFFFFFu32).wrapping_mul(2);
    assert_eq!(read_reg(&state, REG_A), result as i32 as i64 as u64);
}

#[test]
fn test_mulw_register_aliasing_rd_eq_rs1() {
    let executor = Rv64MulWExecutor::new(Rv64MulWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // rd == rs1: should read rs1 before writing rd
    write_reg(&mut state, REG_A, 7);
    write_reg(&mut state, REG_C, 6);
    let inst = make_instruction(REG_A, REG_A, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 42);
}

#[test]
#[should_panic]
fn test_mulw_invalid_instruction_rejected() {
    let executor = Rv64MulWExecutor::new(Rv64MulWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Invalid: d must be RV32_REGISTER_AS.
    let inst = Instruction::from_usize::<7>(
        VmOpcode::from_usize(Rv64MulWOpcode::MULW.global_opcode_usize()),
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
