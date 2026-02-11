use crate::test_utils::{create_exec_state, execute_instruction, read_reg, write_reg};
use openvm_circuit::{
    arch::{execution_mode::ExecutionCtx, VmExecState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
    VmOpcode,
};
use openvm_rv64im_transpiler::Rv64MulHOpcode;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use crate::Rv64MulHExecutor;

type F = BabyBear;

const REG_A: u32 = 0;
const REG_B: u32 = 8;
const REG_C: u32 = 16;
const START_PC: u32 = 0x1000;

fn make_instruction(opcode: Rv64MulHOpcode, rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
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
    executor: &Rv64MulHExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64MulHOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}

// ---------------------------------------------------------------------------
// MULH (signed × signed, upper 64 bits)
// ---------------------------------------------------------------------------

#[test]
fn test_mulh_small_values() {
    let executor = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 2);
    write_reg(&mut state, REG_C, 3);
    let inst = make_instruction(Rv64MulHOpcode::MULH, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // 2 * 3 = 6, upper 64 bits = 0
    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_mulh_negative_result() {
    let executor = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // (-1) * 1 = -1 in 128-bit signed, upper 64 = 0xFFFFFFFFFFFFFFFF
    write_reg(&mut state, REG_B, u64::MAX); // -1
    write_reg(&mut state, REG_C, 1);
    let inst = make_instruction(Rv64MulHOpcode::MULH, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), u64::MAX);
}

#[test]
fn test_mulh_large_product() {
    let executor = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // 2^63 * 2 as signed: i64::MIN * 2 = -2^64, upper 64 bits = -1
    write_reg(&mut state, REG_B, i64::MIN as u64);
    write_reg(&mut state, REG_C, 2);
    let inst = make_instruction(Rv64MulHOpcode::MULH, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), u64::MAX);
}

// ---------------------------------------------------------------------------
// MULHU (unsigned × unsigned, upper 64 bits)
// ---------------------------------------------------------------------------

#[test]
fn test_mulhu_small() {
    let executor = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 100);
    write_reg(&mut state, REG_C, 200);
    let inst = make_instruction(Rv64MulHOpcode::MULHU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_mulhu_max() {
    let executor = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // u64::MAX * u64::MAX = (2^64-1)^2 = 2^128 - 2*2^64 + 1
    // upper 64 bits = 2^64 - 2 = 0xFFFFFFFFFFFFFFFE
    write_reg(&mut state, REG_B, u64::MAX);
    write_reg(&mut state, REG_C, u64::MAX);
    let inst = make_instruction(Rv64MulHOpcode::MULHU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), u64::MAX - 1);
}

// ---------------------------------------------------------------------------
// MULHSU (signed × unsigned, upper 64 bits)
// ---------------------------------------------------------------------------

#[test]
fn test_mulhsu_positive() {
    let executor = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, u64::MAX);
    let inst = make_instruction(Rv64MulHOpcode::MULHSU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    // 1 * 0xFFFFFFFFFFFFFFFF = 0xFFFFFFFFFFFFFFFF, upper 64 = 0
    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_mulhsu_negative_rs1() {
    let executor = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // rs1 = -1 (signed), rs2 = 1 (unsigned)
    // (-1) * 1 = -1, upper 64 bits = 0xFFFFFFFFFFFFFFFF
    write_reg(&mut state, REG_B, u64::MAX); // -1 as signed
    write_reg(&mut state, REG_C, 1);
    let inst = make_instruction(Rv64MulHOpcode::MULHSU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), u64::MAX);
}

#[test]
fn test_mulh_advances_pc() {
    let executor = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, 1);
    write_reg(&mut state, REG_C, 1);
    let inst = make_instruction(Rv64MulHOpcode::MULH, REG_A, REG_B, REG_C);

    let pc = execute(&executor, &mut state, &inst);
    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_mulh_negative_times_negative() {
    let executor = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // (-2) * (-3) = 6. Product fits in lower 64 bits, upper 64 = 0
    write_reg(&mut state, REG_B, (-2i64) as u64);
    write_reg(&mut state, REG_C, (-3i64) as u64);
    let inst = make_instruction(Rv64MulHOpcode::MULH, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_mulhu_with_zero() {
    let executor = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_B, u64::MAX);
    write_reg(&mut state, REG_C, 0);
    let inst = make_instruction(Rv64MulHOpcode::MULHU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
fn test_mulhsu_large_values() {
    let executor = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // rs1 = -1 (signed), rs2 = u64::MAX (unsigned)
    // (-1) * (2^64-1) = -(2^64-1) = -2^64 + 1
    // In 128-bit signed: 0xFFFFFFFFFFFFFFFF_0000000000000001
    // Upper 64 bits: 0xFFFFFFFFFFFFFFFF
    write_reg(&mut state, REG_B, u64::MAX); // -1 as signed
    write_reg(&mut state, REG_C, u64::MAX);
    let inst = make_instruction(Rv64MulHOpcode::MULHSU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), u64::MAX);
}

#[test]
fn test_mulh_i64_min_times_i64_min() {
    let executor = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // i64::MIN * i64::MIN = (-2^63)^2 = 2^126
    // Full 128-bit: 0x40000000_00000000_00000000_00000000
    // Upper 64 bits: 0x4000000000000000
    write_reg(&mut state, REG_B, i64::MIN as u64);
    write_reg(&mut state, REG_C, i64::MIN as u64);
    let inst = make_instruction(Rv64MulHOpcode::MULH, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0x4000000000000000);
}

#[test]
fn test_mulhu_one() {
    let executor = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // u64::MAX * 1 fits in 64 bits, upper 64 = 0
    write_reg(&mut state, REG_B, u64::MAX);
    write_reg(&mut state, REG_C, 1);
    let inst = make_instruction(Rv64MulHOpcode::MULHU, REG_A, REG_B, REG_C);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_A), 0);
}

#[test]
#[should_panic]
fn test_mulh_invalid_instruction_rejected() {
    let executor = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Invalid: d must be RV32_REGISTER_AS.
    let inst = Instruction::from_usize::<7>(
        VmOpcode::from_usize(Rv64MulHOpcode::MULH.global_opcode_usize()),
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
