use openvm_circuit::arch::BLOCK_FE_WIDTH;
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, LOADBU, LOADD, LOADHU, LOADWU};

use crate::adapters::{
    rv64_bytes_to_u16_block, rv64_u16_block_to_bytes, BYTE_ACCESS_WIDTH, DOUBLEWORD_ACCESS_WIDTH,
    HALFWORD_ACCESS_WIDTH, WORD_ACCESS_WIDTH,
};

#[derive(Clone, Copy, derive_new::new)]
pub struct LoadExecutor<const LOAD_WIDTH: usize> {
    pub offset: usize,
}

pub(crate) fn load_byte_write_data(
    read_data: [u16; BLOCK_FE_WIDTH],
    byte_shift: usize,
) -> [u16; BLOCK_FE_WIDTH] {
    debug_assert!(byte_shift < 2 * BLOCK_FE_WIDTH);
    let bytes = rv64_u16_block_to_bytes(read_data);
    let mut loaded = [0u8; 2 * BLOCK_FE_WIDTH];
    loaded[0] = bytes[byte_shift];
    rv64_bytes_to_u16_block(loaded)
}

/// Returns the register write data for an unsigned load at any byte shift, including accesses
/// that span both blocks.
pub(crate) fn load_write_data(
    opcode: Rv64LoadStoreOpcode,
    read_data: [[u16; BLOCK_FE_WIDTH]; 2],
    byte_shift: usize,
) -> [u16; BLOCK_FE_WIDTH] {
    debug_assert!(byte_shift < 2 * BLOCK_FE_WIDTH);
    let width = load_width_for_opcode(opcode);
    let mut bytes = [0u8; 4 * BLOCK_FE_WIDTH];
    bytes[..2 * BLOCK_FE_WIDTH].copy_from_slice(&rv64_u16_block_to_bytes(read_data[0]));
    bytes[2 * BLOCK_FE_WIDTH..].copy_from_slice(&rv64_u16_block_to_bytes(read_data[1]));
    let mut loaded = [0u8; 2 * BLOCK_FE_WIDTH];
    loaded[..width].copy_from_slice(&bytes[byte_shift..byte_shift + width]);
    rv64_bytes_to_u16_block(loaded)
}

pub(crate) fn load_width_for_opcode(opcode: Rv64LoadStoreOpcode) -> usize {
    match opcode {
        LOADD => DOUBLEWORD_ACCESS_WIDTH,
        LOADWU => WORD_ACCESS_WIDTH,
        LOADHU => HALFWORD_ACCESS_WIDTH,
        LOADBU => BYTE_ACCESS_WIDTH,
        _ => unreachable!("unsupported unsigned load opcode: {opcode:?}"),
    }
}
