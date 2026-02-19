use openvm_circuit::{
    arch::{execution_mode::ExecutionCtx, VmExecState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode, VmOpcode,
};
use openvm_rv64im_transpiler::Rv64AuipcOpcode;
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use crate::{
    test_utils::{create_exec_state, execute_instruction, read_reg},
    Rv64AuipcExecutor,
};

type F = BabyBear;

const REG_A: u32 = 0;
const START_PC: u32 = 0x1000;

fn make_auipc_instruction(rd: u32, imm_shifted: u32) -> Instruction<F> {
    // imm_shifted is ((imm & 0xfffff000) >> 8) — the value stored in c
    Instruction::new(
        VmOpcode::from_usize(Rv64AuipcOpcode::AUIPC.global_opcode_usize()),
        F::from_canonical_u32(rd),
        F::ZERO,
        F::from_canonical_u32(imm_shifted),
        F::ONE,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    )
}

fn execute(
    executor: &Rv64AuipcExecutor,
    state: &mut VmExecState<F, GuestMemory, ExecutionCtx>,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(
        executor,
        Rv64AuipcOpcode::iter().map(|x| x.global_opcode()),
        state,
        inst,
        START_PC,
    )
}

#[test]
fn test_auipc_basic() {
    let executor = Rv64AuipcExecutor::new(Rv64AuipcOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // AUIPC with upper 20 bits = 1 → offset = 0x1000
    // c = (0x1000 >> 8) = 0x10
    let inst = make_auipc_instruction(REG_A, 0x10);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
    // rd = pc + 0x1000 = 0x1000 + 0x1000 = 0x2000
    assert_eq!(read_reg(&mut state, REG_A), 0x2000);
}

#[test]
fn test_auipc_sign_extends() {
    let executor = Rv64AuipcExecutor::new(Rv64AuipcOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // AUIPC with upper 20 bits = 0x80000 → offset = 0x80000000
    // c = (0x80000000 >> 8) = 0x800000
    let inst = make_auipc_instruction(REG_A, 0x800000);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
    // rd = pc + 0x80000000 = 0x1000 + 0x80000000 = 0x80001000
    // Bit 31 is set, so sign-extends to 0xFFFFFFFF_80001000
    assert_eq!(read_reg(&mut state, REG_A), 0xFFFFFFFF_80001000u64);
}

#[test]
fn test_auipc_advances_pc() {
    let executor = Rv64AuipcExecutor::new(Rv64AuipcOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let inst = make_auipc_instruction(REG_A, 0);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
    // rd = pc + 0 = 0x1000
    assert_eq!(read_reg(&mut state, REG_A), START_PC as u64);
}

#[test]
fn test_auipc_large_positive() {
    let executor = Rv64AuipcExecutor::new(Rv64AuipcOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // upper 20 bits = 0x7FFFF → offset = 0x7FFFF000
    // c = (0x7FFFF000 >> 8) = 0x7FFFF0
    let inst = make_auipc_instruction(REG_A, 0x7FFFF0);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
    // rd = 0x1000 + 0x7FFFF000 = 0x80000000
    // Bit 31 is set → sign-extends to 0xFFFFFFFF_80000000
    assert_eq!(read_reg(&mut state, REG_A), 0xFFFFFFFF_80000000u64);
}

#[test]
fn test_auipc_max_imm20() {
    let executor = Rv64AuipcExecutor::new(Rv64AuipcOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // upper 20 bits = 0xFFFFF → offset = 0xFFFFF000
    // c = (0xFFFFF000 >> 8) = 0xFFFFF0
    let inst = make_auipc_instruction(REG_A, 0xFFFFF0);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, START_PC + DEFAULT_PC_STEP);
    // rd = 0x1000 + 0xFFFFF000 = 0x100000000, truncated to 32 bits = 0x00000000
    // sign-extended: 0x0000_0000_0000_0000
    let expected = (START_PC as u64).wrapping_add(0xFFFFF000u64);
    let expected_32 = expected as u32;
    assert_eq!(
        read_reg(&mut state, REG_A),
        expected_32 as i32 as i64 as u64
    );
}

// In TCO mode the program array is sized by (pc - pc_base) / step.
// high_pc = 0xFFFFF000 with pc_base = 0x1000 would allocate ~1 billion entries.
#[cfg(not(feature = "tco"))]
#[test]
fn test_auipc_pc_offset_wrapping() {
    let executor = Rv64AuipcExecutor::new(Rv64AuipcOpcode::CLASS_OFFSET);
    // Use a high PC so adding offset wraps past u32::MAX
    let high_pc: u32 = 0xFFFFF000;
    let mut state = create_exec_state(high_pc);

    // offset = 0x1000 → pc + offset = 0x100000000 wraps to 0x00000000
    let inst = make_auipc_instruction(REG_A, 0x10);
    let pc = execute(&executor, &mut state, &inst);

    assert_eq!(pc, high_pc + DEFAULT_PC_STEP);
    // 0xFFFFF000 + 0x1000 = 0x100000000, truncated to 0x00000000
    assert_eq!(read_reg(&mut state, REG_A), 0);
}

#[test]
#[should_panic]
fn test_auipc_invalid_instruction_rejected() {
    let executor = Rv64AuipcExecutor::new(Rv64AuipcOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let inst = Instruction::new(
        VmOpcode::from_usize(Rv64AuipcOpcode::AUIPC.global_opcode_usize()),
        F::from_canonical_u32(REG_A),
        F::ZERO,
        F::from_canonical_u32(0x10),
        F::ZERO, // invalid d (must be RV32_REGISTER_AS)
        F::ZERO,
        F::ZERO,
        F::ZERO,
    );
    let _ = execute(&executor, &mut state, &inst);
}
