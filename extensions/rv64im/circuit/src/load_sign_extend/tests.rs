use crate::test_utils::{
    create_exec_state, execute_instruction, read_reg, write_mem, write_reg, DATA_MEM_AS,
};
use openvm_circuit::arch::ExecutionError;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
    VmOpcode,
};
use openvm_rv64im_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::Rv64LoadSignExtendExecutor;

type F = BabyBear;

// Note: rd=0 (x0) means f=0 (disabled) in load encoding, so use x1+ for actual loads.
const REG_RD: u32 = 8; // x1
const REG_RS1: u32 = 16; // x2

const START_PC: u32 = 0x1000;

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

fn sign_extend_opcodes() -> impl Iterator<Item = VmOpcode> {
    [LOADB, LOADH, LOADW].into_iter().map(|x| x.global_opcode())
}

fn execute(
    executor: &Rv64LoadSignExtendExecutor,
    state: &mut openvm_circuit::arch::VmExecState<
        F,
        openvm_circuit::system::memory::online::GuestMemory,
        openvm_circuit::arch::execution_mode::ExecutionCtx,
    >,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(executor, sign_extend_opcodes(), state, inst, START_PC)
}

// ---------------------------------------------------------------------------
// LOADB (sign-extending byte) tests
// ---------------------------------------------------------------------------

#[test]
fn test_loadb_positive() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let mut data = [0u8; 8];
    data[0] = 0x42; // positive byte
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADB, REG_RD, REG_RS1, 0);
    let new_pc = execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0x0000_0000_0000_0042);
    assert_eq!(new_pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_loadb_negative() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let mut data = [0u8; 8];
    data[0] = 0x80; // -128 as i8
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADB, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0xFFFF_FFFF_FFFF_FF80);
}

#[test]
fn test_loadb_minus_one() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let mut data = [0u8; 8];
    data[0] = 0xFF; // -1 as i8
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADB, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0xFFFF_FFFF_FFFF_FFFF);
}

#[test]
fn test_loadb_various_shifts() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);

    for shift in 0..8u32 {
        let mut state = create_exec_state(START_PC);
        let addr: u32 = 0x100;
        let mut data = [0u8; 8];
        data[shift as usize] = 0xFE; // -2 as i8
        write_mem(&mut state, addr, data);
        write_reg(&mut state, REG_RS1, (addr + shift) as u64);

        let inst = make_load_instruction(LOADB, REG_RD, REG_RS1, 0);
        execute(&executor, &mut state, &inst);

        assert_eq!(
            read_reg(&state, REG_RD),
            0xFFFF_FFFF_FFFF_FFFE,
            "LOADB sign-extend at shift={shift}"
        );
    }
}

// ---------------------------------------------------------------------------
// LOADH (sign-extending halfword) tests
// ---------------------------------------------------------------------------

#[test]
fn test_loadh_positive() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let mut data = [0u8; 8];
    data[0] = 0x34;
    data[1] = 0x12; // 0x1234, positive
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADH, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0x0000_0000_0000_1234);
}

#[test]
fn test_loadh_negative() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let mut data = [0u8; 8];
    data[0] = 0x00;
    data[1] = 0x80; // 0x8000, negative halfword
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADH, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0xFFFF_FFFF_FFFF_8000);
}

#[test]
fn test_loadh_i16_min() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let val = (i16::MIN as u16).to_le_bytes();
    let mut data = [0u8; 8];
    data[0] = val[0];
    data[1] = val[1];
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADH, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), (i16::MIN as i64) as u64);
}

#[test]
fn test_loadh_shift2() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let mut data = [0u8; 8];
    data[2] = 0xCD;
    data[3] = 0xAB; // 0xABCD at offset 2
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, (addr + 2) as u64);

    let inst = make_load_instruction(LOADH, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0xFFFF_FFFF_FFFF_ABCD);
}

#[test]
fn test_loadh_shift4_and_shift6() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);

    for shift in [4u32, 6] {
        let mut state = create_exec_state(START_PC);
        let addr: u32 = 0x100;
        let mut data = [0u8; 8];
        data[shift as usize] = 0xFF;
        data[shift as usize + 1] = 0x7F; // 0x7FFF, positive
        write_mem(&mut state, addr, data);
        write_reg(&mut state, REG_RS1, (addr + shift) as u64);

        let inst = make_load_instruction(LOADH, REG_RD, REG_RS1, 0);
        execute(&executor, &mut state, &inst);

        assert_eq!(
            read_reg(&state, REG_RD),
            0x0000_0000_0000_7FFF,
            "LOADH at shift={shift}"
        );
    }
}

// ---------------------------------------------------------------------------
// LOADW (sign-extending word — new for RV64) tests
// ---------------------------------------------------------------------------

#[test]
fn test_loadw_positive() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let val = 0x12345678u32;
    let mut data = [0u8; 8];
    data[..4].copy_from_slice(&val.to_le_bytes());
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADW, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0x0000_0000_1234_5678);
}

#[test]
fn test_loadw_negative() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let val = 0x8000_0000u32; // i32::MIN
    let mut data = [0u8; 8];
    data[..4].copy_from_slice(&val.to_le_bytes());
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADW, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0xFFFF_FFFF_8000_0000);
}

#[test]
fn test_loadw_all_ones() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let val = 0xFFFF_FFFFu32; // -1 as i32
    let mut data = [0u8; 8];
    data[..4].copy_from_slice(&val.to_le_bytes());
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADW, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0xFFFF_FFFF_FFFF_FFFF);
}

#[test]
fn test_loadw_shift4() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let val = 0xDEAD_BEEFu32;
    let mut data = [0u8; 8];
    data[4..8].copy_from_slice(&val.to_le_bytes());
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, (addr + 4) as u64);

    let inst = make_load_instruction(LOADW, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    // 0xDEADBEEF has high bit set → sign-extends
    assert_eq!(read_reg(&state, REG_RD), 0xFFFF_FFFF_DEAD_BEEF);
}

// ---------------------------------------------------------------------------
// Disabled load (f=0): no-op
// ---------------------------------------------------------------------------

#[test]
fn test_disabled_load_noop() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let data = [0xFF; 8];
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    // rd=0, f=0 (disabled)
    let inst = make_load_instruction(LOADB, 0, REG_RS1, 0);
    let new_pc = execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, 0), 0);
    assert_eq!(new_pc, START_PC + DEFAULT_PC_STEP);
}

// ---------------------------------------------------------------------------
// Edge cases: i8::MIN, i16::MIN, i32::MIN
// ---------------------------------------------------------------------------

#[test]
fn test_loadb_i8_min() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let mut data = [0u8; 8];
    data[0] = i8::MIN as u8; // 0x80
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADB, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), (i8::MIN as i64) as u64);
}

#[test]
fn test_loadw_i32_min() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    let mut data = [0u8; 8];
    data[..4].copy_from_slice(&(i32::MIN as u32).to_le_bytes());
    write_mem(&mut state, addr, data);
    write_reg(&mut state, REG_RS1, addr as u64);

    let inst = make_load_instruction(LOADW, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), (i32::MIN as i64) as u64);
}

// ---------------------------------------------------------------------------
// Load with immediate offset
// ---------------------------------------------------------------------------

#[test]
fn test_loadb_with_positive_imm() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let base_addr: u32 = 0x100;
    let target_addr: u32 = 0x108;
    let mut data = [0u8; 8];
    data[0] = 0x80; // -128
    write_mem(&mut state, target_addr, data);
    write_reg(&mut state, REG_RS1, base_addr as u64);

    let inst = make_load_instruction(LOADB, REG_RD, REG_RS1, 8);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0xFFFF_FFFF_FFFF_FF80);
}

#[test]
fn test_loadh_invalid_odd_shift_sets_error() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    write_mem(&mut state, addr, [0u8; 8]);
    write_reg(&mut state, REG_RS1, (addr + 1) as u64); // odd shift -> invalid for LOADH

    let inst = make_load_instruction(LOADH, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert!(matches!(state.exit_code, Err(ExecutionError::Fail { .. })));
}

#[test]
fn test_loadw_invalid_shift_sets_error() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let addr: u32 = 0x100;
    write_mem(&mut state, addr, [0u8; 8]);
    write_reg(&mut state, REG_RS1, (addr + 2) as u64); // shift=2 -> invalid for LOADW

    let inst = make_load_instruction(LOADW, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert!(matches!(state.exit_code, Err(ExecutionError::Fail { .. })));
}

#[test]
fn test_load_sign_extend_uses_lower_32_bits_of_rs1() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let low_addr: u32 = 0x180;
    let mut data = [0u8; 8];
    data[0] = 0x80; // -128 as i8
    write_mem(&mut state, low_addr, data);
    write_reg(
        &mut state,
        REG_RS1,
        (0x1234_5678u64 << 32) | low_addr as u64,
    );

    let inst = make_load_instruction(LOADB, REG_RD, REG_RS1, 0);
    execute(&executor, &mut state, &inst);

    assert_eq!(read_reg(&state, REG_RD), 0xFFFF_FFFF_FFFF_FF80);
}

#[test]
#[should_panic]
fn test_load_sign_extend_invalid_instruction_rejected() {
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // Invalid: d must be RV32_REGISTER_AS.
    let inst = Instruction::from_usize::<7>(
        VmOpcode::from_usize(LOADB.global_opcode_usize()),
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
