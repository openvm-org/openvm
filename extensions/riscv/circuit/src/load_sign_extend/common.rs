use openvm_circuit::arch::BLOCK_FE_WIDTH;
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, LOADB, LOADH, LOADW};

use crate::adapters::{
    rv64_bytes_to_u16_block, rv64_u16_block_to_bytes, BYTE_ACCESS_WIDTH, HALFWORD_ACCESS_WIDTH,
    WORD_ACCESS_WIDTH,
};

#[derive(Clone, Copy, derive_new::new)]
pub struct LoadSignExtendExecutor<const LOAD_WIDTH: usize, const NUM_BLOCKS: usize = 2> {
    pub offset: usize,
}

/// Returns the register write data for a signed load at any byte shift, including accesses that
/// span both blocks.
pub(crate) fn load_sign_extend_write_data(
    opcode: Rv64LoadStoreOpcode,
    read_data: [[u16; BLOCK_FE_WIDTH]; 2],
    byte_shift: usize,
) -> [u16; BLOCK_FE_WIDTH] {
    debug_assert!(byte_shift < 2 * BLOCK_FE_WIDTH);
    let width = load_sign_extend_width_for_opcode(opcode);
    let mut bytes = [0u8; 4 * BLOCK_FE_WIDTH];
    bytes[..2 * BLOCK_FE_WIDTH].copy_from_slice(&rv64_u16_block_to_bytes(read_data[0]));
    bytes[2 * BLOCK_FE_WIDTH..].copy_from_slice(&rv64_u16_block_to_bytes(read_data[1]));
    let sign = (bytes[byte_shift + width - 1] as i8) < 0;
    let mut loaded = [if sign { 0xff } else { 0 }; 2 * BLOCK_FE_WIDTH];
    loaded[..width].copy_from_slice(&bytes[byte_shift..byte_shift + width]);
    rv64_bytes_to_u16_block(loaded)
}

pub(crate) fn load_sign_extend_width_for_opcode(opcode: Rv64LoadStoreOpcode) -> usize {
    match opcode {
        LOADW => WORD_ACCESS_WIDTH,
        LOADH => HALFWORD_ACCESS_WIDTH,
        LOADB => BYTE_ACCESS_WIDTH,
        _ => unreachable!("unsupported signed load opcode: {opcode:?}"),
    }
}
