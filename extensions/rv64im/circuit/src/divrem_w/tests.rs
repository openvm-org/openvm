use openvm_circuit::{
    arch::{execution_mode::ExecutionCtx, InterpreterExecutor, VmExecState, VmState},
    system::memory::online::{AddressMap, GuestMemory},
};
use crate::test_utils::{create_exec_state, execute_instruction, read_reg, write_reg};
use strum::IntoEnumIterator;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::RV32_REGISTER_AS,
    LocalOpcode, VmOpcode,
};
use openvm_rv64im_transpiler::Rv64DivRemWOpcode;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::Rv64DivRemWExecutor;

type F = BabyBear;

const REG_A: u32 = 0;
const REG_B: u32 = 8;
const REG_C: u32 = 16;
const START_PC: u32 = 0x1000;


fn make_instruction(opcode: Rv64DivRemWOpcode, rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
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
    executor: &Rv64DivRemWExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64DivRemWOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}


#[test]
fn test_divw_basic() {
    let executor = Rv64DivRemWExecutor::new(Rv64DivRemWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 7);
    let inst = make_instruction(Rv64DivRemWOpcode::DIVW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 6);
}

#[test]
fn test_divw_by_zero() {
    let executor = Rv64DivRemWExecutor::new(Rv64DivRemWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 0);
    let inst = make_instruction(Rv64DivRemWOpcode::DIVW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // 0xFFFFFFFF sign-extended to 0xFFFFFFFFFFFFFFFF
    assert_eq!(read_reg(&state, REG_A), u64::MAX);
}

#[test]
fn test_divw_overflow() {
    let executor = Rv64DivRemWExecutor::new(Rv64DivRemWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // i32::MIN / -1 = i32::MIN (overflow)
    write_reg(&mut state, REG_B, 0x80000000); // i32::MIN
    write_reg(&mut state, REG_C, 0xFFFFFFFF); // -1 as u32
    let inst = make_instruction(Rv64DivRemWOpcode::DIVW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // i32::MIN sign-extended
    assert_eq!(read_reg(&state, REG_A), 0xFFFFFFFF80000000);
}

#[test]
fn test_divuw_basic() {
    let executor = Rv64DivRemWExecutor::new(Rv64DivRemWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 100);
    write_reg(&mut state, REG_C, 10);
    let inst = make_instruction(Rv64DivRemWOpcode::DIVUW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 10);
}

#[test]
fn test_remw_basic() {
    let executor = Rv64DivRemWExecutor::new(Rv64DivRemWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 43);
    write_reg(&mut state, REG_C, 7);
    let inst = make_instruction(Rv64DivRemWOpcode::REMW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 1);
}

#[test]
fn test_remw_by_zero() {
    let executor = Rv64DivRemWExecutor::new(Rv64DivRemWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 0);
    let inst = make_instruction(Rv64DivRemWOpcode::REMW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 42);
}

#[test]
fn test_remuw_basic() {
    let executor = Rv64DivRemWExecutor::new(Rv64DivRemWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 103);
    write_reg(&mut state, REG_C, 10);
    let inst = make_instruction(Rv64DivRemWOpcode::REMUW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 3);
}

#[test]
fn test_divrem_w_advances_pc() {
    let executor = Rv64DivRemWExecutor::new(Rv64DivRemWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 10);
    write_reg(&mut state, REG_C, 3);
    let inst = make_instruction(Rv64DivRemWOpcode::DIVW, REG_A, REG_B, REG_C);

    let pc = execute(&executor, &mut state, &inst);
    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_divuw_by_zero() {
    let executor = Rv64DivRemWExecutor::new(Rv64DivRemWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 0);
    let inst = make_instruction(Rv64DivRemWOpcode::DIVUW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // 0xFFFFFFFF sign-extended to 0xFFFFFFFFFFFFFFFF
    assert_eq!(read_reg(&state, REG_A), u64::MAX);
}

#[test]
fn test_remuw_by_zero() {
    let executor = Rv64DivRemWExecutor::new(Rv64DivRemWOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 42);
    write_reg(&mut state, REG_C, 0);
    let inst = make_instruction(Rv64DivRemWOpcode::REMUW, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // Returns dividend (lower 32 bits), sign-extended
    assert_eq!(read_reg(&state, REG_A), 42);
}