use openvm_circuit::arch::ExecutionError;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
    VmOpcode,
};
use openvm_rv64im_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use crate::{
    test_utils::{
        create_exec_state, execute_instruction, read_mem, read_reg, write_mem, write_reg,
        DATA_MEM_AS,
    },
    Rv64LoadStoreExecutor,
};

type F = BabyBear;

// Register byte offsets (RV64: 8 bytes per register)
// Note: rd=0 (x0) means f=0 (disabled) in load encoding, so use x1+ for actual loads.
const REG_RD: u32 = 8; // rd (x1)
const REG_RS1: u32 = 16; // rs1 (x2, base address)
const REG_RS2: u32 = 24; // rs2 (x3, store value source)

const START_PC: u32 = 0x1000;

/// Build a load instruction:
///   a = rd byte offset, b = rs1 byte offset
///   c = lower 16 bits of imm, d = 1 (RV32_REGISTER_AS), e = 2 (data mem AS)
///   f = enabled (rd != 0), g = sign bit of imm
fn make_load_instruction(
    opcode: Rv64LoadStoreOpcode,
    rd: u32,
    rs1: u32,
    imm: i32,
) -> Instruction<F> {
    let imm_u32 = imm as u32;
    Instruction::from_usize::<7>(
        VmOpcode::from_usize(opcode.global_opcode_usize()),
        [
            rd as usize,
            rs1 as usize,
            (imm_u32 & 0xffff) as usize,
            RV32_REGISTER_AS as usize,
            DATA_MEM_AS as usize,
            if rd != 0 { 1 } else { 0 },
            if imm < 0 { 1 } else { 0 },
        ],
    )
}

/// Build a store instruction:
///   a = rs2 byte offset, b = rs1 byte offset
///   c = lower 16 bits of imm, d = 1, e = 2 (data mem AS)
///   f = 1 (always enabled), g = sign bit of imm
fn make_store_instruction(
    opcode: Rv64LoadStoreOpcode,
    rs2: u32,
    rs1: u32,
    imm: i32,
) -> Instruction<F> {
    let imm_u32 = imm as u32;
    Instruction::from_usize::<7>(
        VmOpcode::from_usize(opcode.global_opcode_usize()),
        [
            rs2 as usize,
            rs1 as usize,
            (imm_u32 & 0xffff) as usize,
            RV32_REGISTER_AS as usize,
            DATA_MEM_AS as usize,
            1,
            if imm < 0 { 1 } else { 0 },
        ],
    )
}

fn loadstore_opcodes() -> impl Iterator<Item = VmOpcode> {
    Rv64LoadStoreOpcode::iter()
        .take(STOREB as usize + 1)
        .map(|x| x.global_opcode())
}

fn execute(
    executor: &Rv64LoadStoreExecutor,
    state: &mut openvm_circuit::arch::VmExecState<
        F,
        openvm_circuit::system::memory::online::GuestMemory,
        openvm_circuit::arch::execution_mode::ExecutionCtx,
    >,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(executor, loadstore_opcodes(), state, inst, START_PC)
}

// ---------------------------------------------------------------------------
// LOADD tests
// ---------------------------------------------------------------------------

#[test]
fn test_loadd() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let data = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADD, REG_RD, REG_RS1, 0);
    let new_pc = execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), u64::from_le_bytes(data));
    assert_eq!(new_pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_loadd_with_offset() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let data = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22];
    write_mem(&mut state, addr, data);
    // rs1 = addr - 8, imm = 8
    write_reg(&mut state, REG_RS1, (addr - 8) as u64);

    let inst = make_load_instruction(LOADD, REG_RD, REG_RS1, 8);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), u64::from_le_bytes(data));
}

// ---------------------------------------------------------------------------
// LOADWU tests
// ---------------------------------------------------------------------------

#[test]
fn test_loadwu_shift0() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let data = [0x11, 0x22, 0x33, 0x44, 0xAA, 0xBB, 0xCC, 0xDD];
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADWU, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    // Should load 4 bytes from shift=0, zero-extend to 8 bytes
    assert_eq!(read_reg(&state, REG_RD), 0x0000_0000_4433_2211);
}

#[test]
fn test_loadwu_shift4() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let data = [0x11, 0x22, 0x33, 0x44, 0xAA, 0xBB, 0xCC, 0xDD];
    write_mem(&mut state, addr, data);
    // Point to addr+4 so that shift_amount = 4, aligned_ptr = addr
    write_reg(&mut state, REG_RS1, (addr + 4) as u64);

    let inst = make_load_instruction(LOADWU, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    // Should load 4 bytes from offset 4: [0xAA, 0xBB, 0xCC, 0xDD], zero-extend
    assert_eq!(read_reg(&state, REG_RD), 0x0000_0000_DDCC_BBAA);
}

// ---------------------------------------------------------------------------
// LOADHU tests
// ---------------------------------------------------------------------------

#[test]
fn test_loadhu_shift0() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let data = [0x34, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADHU, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0x1234);
}

#[test]
fn test_loadhu_shift2() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let data = [0x00, 0x00, 0xCD, 0xAB, 0x00, 0x00, 0x00, 0x00];
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, (addr + 2) as u64);

    let inst = make_load_instruction(LOADHU, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0xABCD);
}

#[test]
fn test_loadhu_shift4() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let data = [0x00, 0x00, 0x00, 0x00, 0xEF, 0xBE, 0x00, 0x00];
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, (addr + 4) as u64);

    let inst = make_load_instruction(LOADHU, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0xBEEF);
}

#[test]
fn test_loadhu_shift6() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFE, 0xCA];
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, (addr + 6) as u64);

    let inst = make_load_instruction(LOADHU, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0xCAFE);
}

// ---------------------------------------------------------------------------
// LOADBU tests
// ---------------------------------------------------------------------------

#[test]
fn test_loadbu_shift0() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let data = [0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADBU, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0x42);
}

#[test]
fn test_loadbu_various_shifts() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);

    for shift in 0..8u32 {
        let mut state = create_exec_state(START_PC);
        let addr: u32 = 0x100;
        let mut data = [0u8; 8];
        data[shift as usize] = 0xA0 + shift as u8;
        write_mem(&mut state, addr, data);
        write_reg(&mut state, REG_RS1, (addr + shift) as u64);

        let inst = make_load_instruction(LOADBU, REG_RD, REG_RS1, 0);
        execute(&executor, &mut state, &inst);

        assert_eq!(read_reg(&state, REG_RD), (0xA0 + shift) as u64);
    }
}

// ---------------------------------------------------------------------------
// STORED tests
// ---------------------------------------------------------------------------

#[test]
fn test_stored() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x200;
    let val: u64 = 0x1122334455667788;
    write_reg(&mut state, REG_RS1, addr as u64);
    write_reg(&mut state, REG_RS2, val);

    let inst = make_store_instruction(STORED, REG_RS2, REG_RS1, 0);
    let new_pc = execute(&executor, &mut state, &inst);

    assert_eq!(read_mem(&state, addr), val.to_le_bytes());
    assert_eq!(new_pc, START_PC + DEFAULT_PC_STEP);
}

// ---------------------------------------------------------------------------
// STOREW tests
// ---------------------------------------------------------------------------

#[test]
fn test_storew_shift0() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x200;
    // Pre-fill memory
    let existing = [0xFF, 0xFF, 0xFF, 0xFF, 0xAA, 0xBB, 0xCC, 0xDD];
    write_mem(&mut state, addr, existing);
    write_reg(&mut state, REG_RS1, addr as u64);
    write_reg(&mut state, REG_RS2, 0x0000_0000_4433_2211);

    let inst = make_store_instruction(STOREW, REG_RS2, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    let result = read_mem(&state, addr);
    // Lower 4 bytes overwritten, upper 4 preserved
    assert_eq!(result, [0x11, 0x22, 0x33, 0x44, 0xAA, 0xBB, 0xCC, 0xDD]);
}

#[test]
fn test_storew_shift4() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x200;
    let existing = [0x11, 0x22, 0x33, 0x44, 0xFF, 0xFF, 0xFF, 0xFF];
    write_mem(&mut state, addr, existing);
    write_reg(&mut state, REG_RS1, (addr + 4) as u64);
    write_reg(&mut state, REG_RS2, 0x0000_0000_DDCC_BBAA);

    let inst = make_store_instruction(STOREW, REG_RS2, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    let result = read_mem(&state, addr);
    // Lower 4 preserved, upper 4 overwritten
    assert_eq!(result, [0x11, 0x22, 0x33, 0x44, 0xAA, 0xBB, 0xCC, 0xDD]);
}

// ---------------------------------------------------------------------------
// STOREH tests
// ---------------------------------------------------------------------------

#[test]
fn test_storeh_shift0() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x200;
    let existing = [0xFF; 8];
    write_mem(&mut state, addr, existing);
    write_reg(&mut state, REG_RS1, addr as u64);
    write_reg(&mut state, REG_RS2, 0x0000_0000_0000_ABCD);

    let inst = make_store_instruction(STOREH, REG_RS2, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    let result = read_mem(&state, addr);
    assert_eq!(result, [0xCD, 0xAB, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
}

#[test]
fn test_storeh_shift2() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x200;
    let existing = [0xFF; 8];
    write_mem(&mut state, addr, existing);
    write_reg(&mut state, REG_RS1, (addr + 2) as u64);
    write_reg(&mut state, REG_RS2, 0x0000_0000_0000_1234);

    let inst = make_store_instruction(STOREH, REG_RS2, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    let result = read_mem(&state, addr);
    assert_eq!(result, [0xFF, 0xFF, 0x34, 0x12, 0xFF, 0xFF, 0xFF, 0xFF]);
}

// ---------------------------------------------------------------------------
// STOREB tests
// ---------------------------------------------------------------------------

#[test]
fn test_storeb() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x200;
    let existing = [0xFF; 8];
    write_mem(&mut state, addr, existing);
    write_reg(&mut state, REG_RS1, (addr + 3) as u64);
    write_reg(&mut state, REG_RS2, 0x0000_0000_0000_0042);

    let inst = make_store_instruction(STOREB, REG_RS2, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    let result = read_mem(&state, addr);
    assert_eq!(result, [0xFF, 0xFF, 0xFF, 0x42, 0xFF, 0xFF, 0xFF, 0xFF]);
}

#[test]
fn test_storeb_various_shifts() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);

    for shift in 0..8u32 {
        let mut state = create_exec_state(START_PC);
        let addr: u32 = 0x200;
        let existing = [0xFF; 8];
        write_mem(&mut state, addr, existing);
        write_reg(&mut state, REG_RS1, (addr + shift) as u64);
        write_reg(&mut state, REG_RS2, 0x42);

        let inst = make_store_instruction(STOREB, REG_RS2, REG_RS1, 0);
        execute(&executor, &mut state, &inst);

        let result = read_mem(&state, addr);
        let mut expected = [0xFF; 8];
        expected[shift as usize] = 0x42;
        assert_eq!(result, expected, "STOREB at shift={shift}");
    }
}

// ---------------------------------------------------------------------------
// Disabled load (f=0, rd=0): no-op
// ---------------------------------------------------------------------------

#[test]
fn test_disabled_load_noop() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let data = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    // rd=0, f=0 (disabled)
    let inst = make_load_instruction(LOADD, 0, REG_RS1, 0);
    let new_pc = execute(&executor, &mut state, &inst);

    // Register 0 should remain 0 (not written)
    assert_eq!(read_reg(&state, 0), 0);
    assert_eq!(new_pc, START_PC + DEFAULT_PC_STEP);
}

// ---------------------------------------------------------------------------
// Load with immediate offset
// ---------------------------------------------------------------------------

#[test]
fn test_loadd_with_positive_imm() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let base_addr: u32 = 0x100;
    let target_addr: u32 = 0x110;
    let data = 0xDEAD_BEEF_CAFE_BABEu64;
    write_mem(&mut state, target_addr, data.to_le_bytes());
    write_reg(&mut state, REG_RS1, base_addr as u64);

    let inst = make_load_instruction(LOADD, REG_RD, REG_RS1, 0x10);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), data);
}

#[test]
fn test_stored_with_positive_imm() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let base_addr: u32 = 0x200;
    let target_addr: u32 = 0x220;
    let val: u64 = 0xAAAA_BBBB_CCCC_DDDD;
    write_reg(&mut state, REG_RS1, base_addr as u64);
    write_reg(&mut state, REG_RS2, val);

    let inst = make_store_instruction(STORED, REG_RS2, REG_RS1, 0x20);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_mem(&state, target_addr), val.to_le_bytes());
}

#[test]
fn test_loadwu_invalid_shift_sets_error() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    write_mem(&mut state, addr, [0u8; 8]);
    write_reg(&mut state, REG_RS1, (addr + 2) as u64); // shift=2 invalid for LOADWU

    let inst = make_load_instruction(LOADWU, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert!(matches!(state.exit_code, Err(ExecutionError::Fail { .. })));
}

#[test]
fn test_loadhu_invalid_shift_sets_error() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    write_mem(&mut state, addr, [0u8; 8]);
    write_reg(&mut state, REG_RS1, (addr + 1) as u64); // odd shift invalid for LOADHU

    let inst = make_load_instruction(LOADHU, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert!(matches!(state.exit_code, Err(ExecutionError::Fail { .. })));
}

#[test]
fn test_storew_invalid_shift_sets_error() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x200;
    write_mem(&mut state, addr, [0u8; 8]);
    write_reg(&mut state, REG_RS1, (addr + 2) as u64); // shift=2 invalid for STOREW
    write_reg(&mut state, REG_RS2, 0x1122_3344);

    let inst = make_store_instruction(STOREW, REG_RS2, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert!(matches!(state.exit_code, Err(ExecutionError::Fail { .. })));
}

#[test]
fn test_storeh_invalid_shift_sets_error() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x200;
    write_mem(&mut state, addr, [0u8; 8]);
    write_reg(&mut state, REG_RS1, (addr + 1) as u64); // odd shift invalid for STOREH
    write_reg(&mut state, REG_RS2, 0xABCD);

    let inst = make_store_instruction(STOREH, REG_RS2, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert!(matches!(state.exit_code, Err(ExecutionError::Fail { .. })));
}

#[test]
fn test_loadstore_uses_lower_32_bits_of_rs1() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let low_addr: u32 = 0x120;
    let data = 0x0123_4567_89AB_CDEFu64;
    write_mem(&mut state, low_addr, data.to_le_bytes());
    write_reg(
        &mut state,
        REG_RS1,
        (0xAAAA_BBBBu64 << 32) | low_addr as u64,
    );

    let inst = make_load_instruction(LOADD, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), data);
}

// ---------------------------------------------------------------------------
// LOADD/STORED alignment error tests (must be 8-byte aligned)
// ---------------------------------------------------------------------------

#[test]
fn test_loadd_unaligned_shift1_sets_error() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    write_mem(&mut state, addr, [0u8; 8]);
    write_reg(&mut state, REG_RS1, (addr + 1) as u64);

    let inst = make_load_instruction(LOADD, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert!(matches!(state.exit_code, Err(ExecutionError::Fail { .. })));
}

#[test]
fn test_loadd_unaligned_shift4_sets_error() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    write_mem(&mut state, addr, [0u8; 8]);
    write_reg(&mut state, REG_RS1, (addr + 4) as u64);

    let inst = make_load_instruction(LOADD, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert!(matches!(state.exit_code, Err(ExecutionError::Fail { .. })));
}

#[test]
fn test_stored_unaligned_shift1_sets_error() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x200;
    write_mem(&mut state, addr, [0u8; 8]);
    write_reg(&mut state, REG_RS1, (addr + 1) as u64);
    write_reg(&mut state, REG_RS2, 0x1234);

    let inst = make_store_instruction(STORED, REG_RS2, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert!(matches!(state.exit_code, Err(ExecutionError::Fail { .. })));
}

#[test]
fn test_stored_unaligned_shift4_sets_error() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x200;
    write_mem(&mut state, addr, [0u8; 8]);
    write_reg(&mut state, REG_RS1, (addr + 4) as u64);
    write_reg(&mut state, REG_RS2, 0x1234);

    let inst = make_store_instruction(STORED, REG_RS2, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert!(matches!(state.exit_code, Err(ExecutionError::Fail { .. })));
}

// ---------------------------------------------------------------------------
// Store-then-load round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_stored_then_loadd_roundtrip() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x300;
    let val: u64 = 0xDEADBEEF_CAFEBABE;
    write_reg(&mut state, REG_RS1, addr as u64);
    write_reg(&mut state, REG_RS2, val);

    // Store
    let store_inst = make_store_instruction(STORED, REG_RS2, REG_RS1, 0);
    execute(&executor, &mut state, &store_inst);

    // Load back into a different register
    let load_inst = make_load_instruction(LOADD, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &load_inst);

    assert_eq!(read_reg(&state, REG_RD), val);
}

#[test]
fn test_storew_then_loadwu_roundtrip() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x300;
    let val: u64 = 0x12345678;
    write_reg(&mut state, REG_RS1, addr as u64);
    write_reg(&mut state, REG_RS2, val);

    let store_inst = make_store_instruction(STOREW, REG_RS2, REG_RS1, 0);
    execute(&executor, &mut state, &store_inst);

    let load_inst = make_load_instruction(LOADWU, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &load_inst);

    assert_eq!(read_reg(&state, REG_RD), val);
}

// ---------------------------------------------------------------------------
// Negative immediate offset
// ---------------------------------------------------------------------------

#[test]
fn test_loadd_with_negative_imm() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let target_addr: u32 = 0x100;
    let data = 0xAAAA_BBBB_CCCC_DDDDu64;
    write_mem(&mut state, target_addr, data.to_le_bytes());
    // rs1 points 8 past target, use imm=-8
    write_reg(&mut state, REG_RS1, (target_addr + 8) as u64);

    let inst = make_load_instruction(LOADD, REG_RD, REG_RS1, -8);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), data);
}

#[test]
fn test_stored_with_negative_imm() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let target_addr: u32 = 0x200;
    let val: u64 = 0x1111_2222_3333_4444;
    write_reg(&mut state, REG_RS1, (target_addr + 16) as u64);
    write_reg(&mut state, REG_RS2, val);

    let inst = make_store_instruction(STORED, REG_RS2, REG_RS1, -16);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_mem(&state, target_addr), val.to_le_bytes());
}

#[test]
#[should_panic]
fn test_loadstore_invalid_instruction_rejected() {
    let executor = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Invalid: d must be RV32_REGISTER_AS.
    let inst = Instruction::from_usize::<7>(
        VmOpcode::from_usize(LOADD.global_opcode_usize()),
        [
            REG_RD as usize,
            REG_RS1 as usize,
            0,
            0,
            DATA_MEM_AS as usize,
            1,
            0,
        ],
    );
    execute(&executor, &mut state, &inst);
}
