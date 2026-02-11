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
use openvm_rv64im_transpiler::Rv64BranchLessThanOpcode;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::Rv64BranchLessThanExecutor;

type F = BabyBear;

const REG_A: u32 = 0;
const REG_B: u32 = 8;
const START_PC: u32 = 0x1000;


fn make_instruction(
    opcode: Rv64BranchLessThanOpcode,
    rs1: u32,
    rs2: u32,
    imm: u32,
) -> Instruction<F> {
    Instruction::from_usize::<7>(
        VmOpcode::from_usize(opcode.global_opcode_usize()),
        [
            rs1 as usize,
            rs2 as usize,
            imm as usize,
            RV32_REGISTER_AS as usize,
            0,
            0,
            0,
        ],
    )
}

fn execute(
    executor: &Rv64BranchLessThanExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64BranchLessThanOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}


#[test]
fn test_blt_less_branches() {
    let executor = Rv64BranchLessThanExecutor::new(Rv64BranchLessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 1);
    write_reg(&mut state, REG_B, 2);
    let inst = make_instruction(Rv64BranchLessThanOpcode::BLT, REG_A, REG_B, 100);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + 100);
}

#[test]
fn test_blt_not_less_falls_through() {
    let executor = Rv64BranchLessThanExecutor::new(Rv64BranchLessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 5);
    write_reg(&mut state, REG_B, 3);
    let inst = make_instruction(Rv64BranchLessThanOpcode::BLT, REG_A, REG_B, 100);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_blt_negative_less_than_positive() {
    let executor = Rv64BranchLessThanExecutor::new(Rv64BranchLessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, u64::MAX); // -1 signed
    write_reg(&mut state, REG_B, 0);
    let inst = make_instruction(Rv64BranchLessThanOpcode::BLT, REG_A, REG_B, 100);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + 100);
}

#[test]
fn test_bltu_unsigned() {
    let executor = Rv64BranchLessThanExecutor::new(Rv64BranchLessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 0);
    write_reg(&mut state, REG_B, u64::MAX);
    let inst = make_instruction(Rv64BranchLessThanOpcode::BLTU, REG_A, REG_B, 100);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + 100);
}

#[test]
fn test_bge_greater_branches() {
    let executor = Rv64BranchLessThanExecutor::new(Rv64BranchLessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 5);
    write_reg(&mut state, REG_B, 3);
    let inst = make_instruction(Rv64BranchLessThanOpcode::BGE, REG_A, REG_B, 100);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + 100);
}

#[test]
fn test_bge_equal_branches() {
    let executor = Rv64BranchLessThanExecutor::new(Rv64BranchLessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 42);
    write_reg(&mut state, REG_B, 42);
    let inst = make_instruction(Rv64BranchLessThanOpcode::BGE, REG_A, REG_B, 100);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + 100);
}

#[test]
fn test_bgeu_unsigned() {
    let executor = Rv64BranchLessThanExecutor::new(Rv64BranchLessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, u64::MAX);
    write_reg(&mut state, REG_B, 0);
    let inst = make_instruction(Rv64BranchLessThanOpcode::BGEU, REG_A, REG_B, 100);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + 100);
}

#[test]
fn test_bltu_not_less_falls_through() {
    let executor = Rv64BranchLessThanExecutor::new(Rv64BranchLessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // u64::MAX >= 0 unsigned, so BLTU should fall through
    write_reg(&mut state, REG_A, u64::MAX);
    write_reg(&mut state, REG_B, 0);
    let inst = make_instruction(Rv64BranchLessThanOpcode::BLTU, REG_A, REG_B, 100);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_bgeu_less_falls_through() {
    let executor = Rv64BranchLessThanExecutor::new(Rv64BranchLessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 0 < u64::MAX unsigned, so BGEU should fall through
    write_reg(&mut state, REG_A, 0);
    write_reg(&mut state, REG_B, u64::MAX);
    let inst = make_instruction(Rv64BranchLessThanOpcode::BGEU, REG_A, REG_B, 100);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_bge_negative_less_than_positive_falls_through() {
    let executor = Rv64BranchLessThanExecutor::new(Rv64BranchLessThanOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // -1 < 0 signed, so BGE should fall through
    write_reg(&mut state, REG_A, u64::MAX); // -1
    write_reg(&mut state, REG_B, 0);
    let inst = make_instruction(Rv64BranchLessThanOpcode::BGE, REG_A, REG_B, 100);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
}