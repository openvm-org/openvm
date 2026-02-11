use crate::test_utils::{
    create_exec_state, execute_instruction, read_mem, write_mem, write_reg, DATA_MEM_AS,
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
    VmOpcode,
};
use openvm_rv64im_transpiler::{
    Rv64HintStoreOpcode::{self, *},
    MAX_HINT_BUFFER_DWORDS,
};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use strum::IntoEnumIterator;

use crate::Rv64HintStoreExecutor;
use openvm_circuit::arch::ExecutionError;

type F = BabyBear;

const REG_NUM_DWORDS: u32 = 8; // x1
const REG_MEM_PTR: u32 = 16; // x2
const START_PC: u32 = 0x1000;

fn make_hint_stored_instruction(mem_ptr_reg: u32) -> Instruction<F> {
    Instruction::from_usize::<5>(
        VmOpcode::from_usize(HINT_STORED.global_opcode_usize()),
        [
            0,
            mem_ptr_reg as usize,
            0,
            RV32_REGISTER_AS as usize,
            DATA_MEM_AS as usize,
        ],
    )
}

fn make_hint_buffer_instruction(num_dwords_reg: u32, mem_ptr_reg: u32) -> Instruction<F> {
    Instruction::from_usize::<5>(
        VmOpcode::from_usize(HINT_BUFFER.global_opcode_usize()),
        [
            num_dwords_reg as usize,
            mem_ptr_reg as usize,
            0,
            RV32_REGISTER_AS as usize,
            DATA_MEM_AS as usize,
        ],
    )
}

fn hintstore_opcodes() -> impl Iterator<Item = VmOpcode> {
    Rv64HintStoreOpcode::iter().map(|x| x.global_opcode())
}

fn execute(
    executor: &Rv64HintStoreExecutor,
    state: &mut openvm_circuit::arch::VmExecState<
        F,
        openvm_circuit::system::memory::online::GuestMemory,
        openvm_circuit::arch::execution_mode::ExecutionCtx,
    >,
    inst: &Instruction<F>,
) -> u32 {
    execute_instruction(executor, hintstore_opcodes(), state, inst, START_PC)
}

fn push_hint_bytes(
    state: &mut openvm_circuit::arch::VmExecState<
        F,
        openvm_circuit::system::memory::online::GuestMemory,
        openvm_circuit::arch::execution_mode::ExecutionCtx,
    >,
    bytes: &[u8],
) {
    state
        .streams
        .hint_stream
        .extend(bytes.iter().copied().map(F::from_canonical_u8));
}

#[test]
fn test_hint_stored() {
    let executor = Rv64HintStoreExecutor::new(Rv64HintStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let mem_ptr: u32 = 0x100;
    write_reg(&mut state, REG_MEM_PTR, mem_ptr as u64);
    let expected = [1u8, 2, 3, 4, 5, 6, 7, 8];
    push_hint_bytes(&mut state, &expected);

    let new_pc = execute(
        &executor,
        &mut state,
        &make_hint_stored_instruction(REG_MEM_PTR),
    );

    assert_eq!(read_mem(&state, mem_ptr), expected);
    assert_eq!(new_pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_hint_buffer_one_dword() {
    let executor = Rv64HintStoreExecutor::new(Rv64HintStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let mem_ptr: u32 = 0x180;
    write_reg(&mut state, REG_NUM_DWORDS, 1);
    write_reg(&mut state, REG_MEM_PTR, mem_ptr as u64);
    let expected = [10u8, 11, 12, 13, 14, 15, 16, 17];
    push_hint_bytes(&mut state, &expected);

    execute(
        &executor,
        &mut state,
        &make_hint_buffer_instruction(REG_NUM_DWORDS, REG_MEM_PTR),
    );

    assert_eq!(read_mem(&state, mem_ptr), expected);
}

#[test]
fn test_hint_buffer_multiple_dwords() {
    let executor = Rv64HintStoreExecutor::new(Rv64HintStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let mem_ptr: u32 = 0x200;
    let num_dwords = 5u64;
    write_reg(&mut state, REG_NUM_DWORDS, num_dwords);
    write_reg(&mut state, REG_MEM_PTR, mem_ptr as u64);

    let mut all_bytes = [0u8; 40];
    for (i, b) in all_bytes.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(3).wrapping_add(1);
    }
    push_hint_bytes(&mut state, &all_bytes);

    execute(
        &executor,
        &mut state,
        &make_hint_buffer_instruction(REG_NUM_DWORDS, REG_MEM_PTR),
    );

    for idx in 0..5u32 {
        let mut expected = [0u8; 8];
        expected.copy_from_slice(&all_bytes[(idx as usize * 8)..((idx as usize + 1) * 8)]);
        assert_eq!(read_mem(&state, mem_ptr + idx * 8), expected);
    }
}

#[test]
fn test_hint_buffer_max_boundary() {
    let executor = Rv64HintStoreExecutor::new(Rv64HintStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let mem_ptr: u32 = 0x300;
    let max = MAX_HINT_BUFFER_DWORDS as u64;
    write_reg(&mut state, REG_NUM_DWORDS, max);
    write_reg(&mut state, REG_MEM_PTR, mem_ptr as u64);

    // Fill exactly 8 * MAX_HINT_BUFFER_DWORDS bytes.
    for i in 0..(MAX_HINT_BUFFER_DWORDS * 8) {
        state
            .streams
            .hint_stream
            .push_back(F::from_canonical_u8((i % 256) as u8));
    }

    let new_pc = execute(
        &executor,
        &mut state,
        &make_hint_buffer_instruction(REG_NUM_DWORDS, REG_MEM_PTR),
    );

    assert_eq!(new_pc, START_PC + DEFAULT_PC_STEP);
    assert_eq!(state.streams.hint_stream.len(), 0);
}

#[test]
fn test_hint_buffer_zero_noop() {
    let executor = Rv64HintStoreExecutor::new(Rv64HintStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let mem_ptr: u32 = 0x380;
    write_reg(&mut state, REG_NUM_DWORDS, 0);
    write_reg(&mut state, REG_MEM_PTR, mem_ptr as u64);

    let sentinel = [0xAAu8; 8];
    write_mem(&mut state, mem_ptr, sentinel);

    let new_pc = execute(
        &executor,
        &mut state,
        &make_hint_buffer_instruction(REG_NUM_DWORDS, REG_MEM_PTR),
    );

    assert_eq!(read_mem(&state, mem_ptr), sentinel);
    assert_eq!(new_pc, START_PC + DEFAULT_PC_STEP);
}

#[test]
fn test_hint_buffer_too_large() {
    let executor = Rv64HintStoreExecutor::new(Rv64HintStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(
        &mut state,
        REG_NUM_DWORDS,
        (MAX_HINT_BUFFER_DWORDS as u64) + 1,
    );
    write_reg(&mut state, REG_MEM_PTR, 0x400);

    execute(
        &executor,
        &mut state,
        &make_hint_buffer_instruction(REG_NUM_DWORDS, REG_MEM_PTR),
    );

    assert!(matches!(
        state.exit_code,
        Err(ExecutionError::HintBufferTooLarge { .. })
    ));
}

#[test]
fn test_hint_buffer_hint_stream_out_of_bounds() {
    let executor = Rv64HintStoreExecutor::new(Rv64HintStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    write_reg(&mut state, REG_NUM_DWORDS, 2);
    write_reg(&mut state, REG_MEM_PTR, 0x420);
    // Need 16 hint bytes for 2 dwords; provide only 15.
    push_hint_bytes(&mut state, &[1u8; 15]);

    execute(
        &executor,
        &mut state,
        &make_hint_buffer_instruction(REG_NUM_DWORDS, REG_MEM_PTR),
    );

    assert!(matches!(
        state.exit_code,
        Err(ExecutionError::HintOutOfBounds { .. })
    ));
}

#[test]
fn test_mem_ptr_uses_lower_32_bits() {
    let executor = Rv64HintStoreExecutor::new(Rv64HintStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    // High 32 bits set; lower 32 bits should be used as address.
    let mem_ptr64 = (0xDEAD_BEEFu64 << 32) | 0x0000_0500u64;
    write_reg(&mut state, REG_MEM_PTR, mem_ptr64);

    let expected = [0x21u8, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28];
    push_hint_bytes(&mut state, &expected);

    execute(
        &executor,
        &mut state,
        &make_hint_stored_instruction(REG_MEM_PTR),
    );

    assert_eq!(read_mem(&state, 0x500), expected);
}

#[test]
fn test_hint_stored_insufficient_data() {
    let executor = Rv64HintStoreExecutor::new(Rv64HintStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let mem_ptr: u32 = 0x100;
    write_reg(&mut state, REG_MEM_PTR, mem_ptr as u64);
    // Need 8 hint bytes for HINT_STORED, provide only 7
    push_hint_bytes(&mut state, &[1u8; 7]);

    execute(
        &executor,
        &mut state,
        &make_hint_stored_instruction(REG_MEM_PTR),
    );

    assert!(matches!(
        state.exit_code,
        Err(ExecutionError::HintOutOfBounds { .. })
    ));
}

#[test]
fn test_hint_stored_consecutive_calls() {
    let executor = Rv64HintStoreExecutor::new(Rv64HintStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let mem_ptr1: u32 = 0x100;
    let mem_ptr2: u32 = 0x108;
    let data1 = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let data2 = [9u8, 10, 11, 12, 13, 14, 15, 16];
    push_hint_bytes(&mut state, &data1);
    push_hint_bytes(&mut state, &data2);

    // First store
    write_reg(&mut state, REG_MEM_PTR, mem_ptr1 as u64);
    execute(
        &executor,
        &mut state,
        &make_hint_stored_instruction(REG_MEM_PTR),
    );
    assert_eq!(read_mem(&state, mem_ptr1), data1);

    // Second store consumes next 8 bytes from hint stream
    write_reg(&mut state, REG_MEM_PTR, mem_ptr2 as u64);
    execute(
        &executor,
        &mut state,
        &make_hint_stored_instruction(REG_MEM_PTR),
    );
    assert_eq!(read_mem(&state, mem_ptr2), data2);

    // Hint stream should be empty
    assert_eq!(state.streams.hint_stream.len(), 0);
}

#[test]
fn test_hint_buffer_num_dwords_from_register() {
    let executor = Rv64HintStoreExecutor::new(Rv64HintStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let mem_ptr: u32 = 0x200;
    // num_dwords has upper 32 bits set, lower 32 = 2
    write_reg(&mut state, REG_NUM_DWORDS, 0xDEAD_0000_0000_0002u64);
    write_reg(&mut state, REG_MEM_PTR, mem_ptr as u64);

    let expected = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    push_hint_bytes(&mut state, &expected);

    execute(
        &executor,
        &mut state,
        &make_hint_buffer_instruction(REG_NUM_DWORDS, REG_MEM_PTR),
    );

    // Should use lower 32 bits of num_dwords (2), reading 16 bytes
    assert_eq!(read_mem(&state, mem_ptr), expected[..8]);
    assert_eq!(read_mem(&state, mem_ptr + 8), expected[8..16]);
}

#[test]
#[should_panic]
fn test_hintstore_invalid_instruction_rejected() {
    let executor = Rv64HintStoreExecutor::new(Rv64HintStoreOpcode::CLASS_OFFSET);
    let mut state = create_exec_state(START_PC);

    let inst = Instruction::from_usize::<5>(
        VmOpcode::from_usize(HINT_STORED.global_opcode_usize()),
        [
            0,
            REG_MEM_PTR as usize,
            0,
            0, // invalid d
            DATA_MEM_AS as usize,
        ],
    );
    let _ = execute(&executor, &mut state, &inst);
}
