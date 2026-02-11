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
use openvm_rv64im_transpiler::Rv64BranchEqualOpcode;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::Rv64BranchEqualExecutor;

type F = BabyBear;

const REG_A: u32 = 0;
const REG_B: u32 = 8;
const START_PC: u32 = 0x1000;


fn make_instruction(
    opcode: Rv64BranchEqualOpcode,
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
    executor: &Rv64BranchEqualExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64BranchEqualOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}


#[test]
fn test_beq_equal_branches() {
    let executor = Rv64BranchEqualExecutor::new(Rv64BranchEqualOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 42);
    write_reg(&mut state, REG_B, 42);
    let inst = make_instruction(Rv64BranchEqualOpcode::BEQ, REG_A, REG_B, 100);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + 100);
}

#[test]
fn test_beq_not_equal_falls_through() {
    let executor = Rv64BranchEqualExecutor::new(Rv64BranchEqualOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 42);
    write_reg(&mut state, REG_B, 43);
    let inst = make_instruction(Rv64BranchEqualOpcode::BEQ, REG_A, REG_B, 100);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_bne_not_equal_branches() {
    let executor = Rv64BranchEqualExecutor::new(Rv64BranchEqualOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 1);
    write_reg(&mut state, REG_B, 2);
    let inst = make_instruction(Rv64BranchEqualOpcode::BNE, REG_A, REG_B, 200);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + 200);
}

#[test]
fn test_bne_equal_falls_through() {
    let executor = Rv64BranchEqualExecutor::new(Rv64BranchEqualOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 42);
    write_reg(&mut state, REG_B, 42);
    let inst = make_instruction(Rv64BranchEqualOpcode::BNE, REG_A, REG_B, 200);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_beq_compares_full_64_bits() {
    let executor = Rv64BranchEqualExecutor::new(Rv64BranchEqualOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Same lower 32 bits, different upper 32 bits
    write_reg(&mut state, REG_A, 0x00000001_00000042);
    write_reg(&mut state, REG_B, 0x00000002_00000042);
    let inst = make_instruction(Rv64BranchEqualOpcode::BEQ, REG_A, REG_B, 100);
    let pc = execute(&executor, &mut state, &inst);

    // Should NOT branch — values differ in upper 32 bits
    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
}