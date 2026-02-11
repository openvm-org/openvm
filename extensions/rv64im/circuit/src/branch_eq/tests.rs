use crate::test_utils::{create_exec_state, execute_instruction, write_reg};
use openvm_circuit::{
    arch::{execution_mode::ExecutionCtx, VmExecState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
    VmOpcode,
};
use openvm_rv64im_transpiler::Rv64BranchEqualOpcode;
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use crate::Rv64BranchEqualExecutor;

type F = BabyBear;

const REG_A: u32 = 0;
const REG_B: u32 = 8;
const START_PC: u32 = 0x1000;

fn make_instruction(opcode: Rv64BranchEqualOpcode, rs1: u32, rs2: u32, imm: u32) -> Instruction<F> {
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

#[test]
fn test_bne_upper_32_bit_difference() {
    let executor = Rv64BranchEqualExecutor::new(Rv64BranchEqualOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Same lower 32 bits, different upper 32 bits → BNE should branch
    write_reg(&mut state, REG_A, 0x00000001_00000042);
    write_reg(&mut state, REG_B, 0x00000002_00000042);
    let inst = make_instruction(Rv64BranchEqualOpcode::BNE, REG_A, REG_B, 200);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + 200);
}

#[test]
fn test_beq_negative_branch_offset() {
    let executor = Rv64BranchEqualExecutor::new(Rv64BranchEqualOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 42);
    write_reg(&mut state, REG_B, 42);
    // Negative offset: use BabyBear field encoding.
    // The c field is interpreted as signed via the field representation.
    // F::ORDER_U32 - 100 encodes -100 in the field.
    let neg_100 = openvm_stark_sdk::p3_baby_bear::BabyBear::ORDER_U32 - 100;
    let inst = make_instruction(Rv64BranchEqualOpcode::BEQ, REG_A, REG_B, neg_100);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC.wrapping_sub(100));
}

#[test]
fn test_beq_both_zero() {
    let executor = Rv64BranchEqualExecutor::new(Rv64BranchEqualOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, 0);
    write_reg(&mut state, REG_B, 0);
    let inst = make_instruction(Rv64BranchEqualOpcode::BEQ, REG_A, REG_B, 100);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + 100);
}

#[test]
fn test_bne_both_max() {
    let executor = Rv64BranchEqualExecutor::new(Rv64BranchEqualOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_A, u64::MAX);
    write_reg(&mut state, REG_B, u64::MAX);
    let inst = make_instruction(Rv64BranchEqualOpcode::BNE, REG_A, REG_B, 200);
    let pc = execute(&executor, &mut state, &inst);

    // Equal → BNE falls through
    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
#[should_panic]
fn test_branch_eq_invalid_instruction_rejected() {
    let executor = Rv64BranchEqualExecutor::new(Rv64BranchEqualOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let inst = Instruction::from_usize::<7>(
        VmOpcode::from_usize(Rv64BranchEqualOpcode::BEQ.global_opcode_usize()),
        [
            REG_A as usize,
            REG_B as usize,
            4,
            0, // invalid d
            0,
            0,
            0,
        ],
    );
    let _ = execute(&executor, &mut state, &inst);
}
