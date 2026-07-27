use openvm_circuit::arch::BLOCK_FE_WIDTH;
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, STOREB, STORED, STOREH, STOREW};

use crate::adapters::{
    rv64_bytes_to_u16_block, rv64_u16_block_to_bytes, BYTE_ACCESS_WIDTH, DOUBLEWORD_ACCESS_WIDTH,
    HALFWORD_ACCESS_WIDTH, WORD_ACCESS_WIDTH,
};

#[derive(Clone, Copy, derive_new::new)]
pub struct StoreExecutor<A, const STORE_WIDTH: usize, const NUM_BLOCKS: usize = 2> {
    adapter: A,
    pub offset: usize,
}

/// Returns the two block values supplied to the adapter for a store at any byte offset. The
/// adapter writes the second block only when the access crosses the first one.
pub(crate) fn store_write_data(
    opcode: Rv64LoadStoreOpcode,
    read_data: [u16; BLOCK_FE_WIDTH],
    prev_data: [[u16; BLOCK_FE_WIDTH]; 2],
    byte_shift: usize,
) -> [[u16; BLOCK_FE_WIDTH]; 2] {
    debug_assert!(byte_shift < 2 * BLOCK_FE_WIDTH);
    let width = store_width_for_opcode(opcode);
    let mut bytes = [0u8; 4 * BLOCK_FE_WIDTH];
    bytes[..2 * BLOCK_FE_WIDTH].copy_from_slice(&rv64_u16_block_to_bytes(prev_data[0]));
    bytes[2 * BLOCK_FE_WIDTH..].copy_from_slice(&rv64_u16_block_to_bytes(prev_data[1]));
    let value = rv64_u16_block_to_bytes(read_data);
    bytes[byte_shift..byte_shift + width].copy_from_slice(&value[..width]);
    [
        rv64_bytes_to_u16_block(bytes[..2 * BLOCK_FE_WIDTH].try_into().unwrap()),
        rv64_bytes_to_u16_block(bytes[2 * BLOCK_FE_WIDTH..].try_into().unwrap()),
    ]
}

pub(crate) fn store_width_for_opcode(opcode: Rv64LoadStoreOpcode) -> usize {
    match opcode {
        STORED => DOUBLEWORD_ACCESS_WIDTH,
        STOREW => WORD_ACCESS_WIDTH,
        STOREH => HALFWORD_ACCESS_WIDTH,
        STOREB => BYTE_ACCESS_WIDTH,
        _ => unreachable!("unsupported store opcode: {opcode:?}"),
    }
}
