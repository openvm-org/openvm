use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{LOADB, LOADH, LOADW};

use super::common::load_sign_extend_write_data;
use crate::adapters::{rv64_bytes_to_u16_block, rv64_u16_block_to_bytes};

#[test]
fn load_sign_extend_sanity_tests() {
    let block1 = rv64_bytes_to_u16_block([156, 92, 17, 203, 44, 118, 240, 5]);

    let read_data = [
        rv64_bytes_to_u16_block([34, 159, 237, 151, 100, 200, 50, 25]),
        block1,
    ];
    assert_eq!(
        load_sign_extend_write_data(LOADH, read_data, 0),
        rv64_bytes_to_u16_block([34, 159, 255, 255, 255, 255, 255, 255])
    );
    assert_eq!(
        load_sign_extend_write_data(LOADH, read_data, 2),
        rv64_bytes_to_u16_block([237, 151, 255, 255, 255, 255, 255, 255])
    );
    assert_eq!(
        load_sign_extend_write_data(LOADH, read_data, 4),
        rv64_bytes_to_u16_block([100, 200, 255, 255, 255, 255, 255, 255])
    );
    assert_eq!(
        load_sign_extend_write_data(LOADH, read_data, 6),
        rv64_bytes_to_u16_block([50, 25, 0, 0, 0, 0, 0, 0])
    );
    // Misaligned within one block, positive.
    assert_eq!(
        load_sign_extend_write_data(LOADH, read_data, 3),
        rv64_bytes_to_u16_block([151, 100, 0, 0, 0, 0, 0, 0])
    );
    // Misaligned across the block boundary, negative sign byte from the second block.
    assert_eq!(
        load_sign_extend_write_data(LOADH, read_data, 7),
        rv64_bytes_to_u16_block([25, 156, 255, 255, 255, 255, 255, 255])
    );

    let read_data = [
        rv64_bytes_to_u16_block([45, 82, 99, 127, 200, 150, 180, 210]),
        block1,
    ];
    for shift in 0..8 {
        let byte = rv64_u16_block_to_bytes(read_data[0])[shift];
        assert_eq!(
            rv64_u16_block_to_bytes(load_sign_extend_write_data(LOADB, read_data, shift)),
            (byte as i8 as i64).to_le_bytes(),
            "LOADB shift={shift}"
        );
    }

    let read_data = [
        rv64_bytes_to_u16_block([0x01, 0x02, 0x03, 0x84, 0xAA, 0xBB, 0xCC, 0xDD]),
        block1,
    ];
    assert_eq!(
        load_sign_extend_write_data(LOADW, read_data, 0),
        rv64_bytes_to_u16_block([0x01, 0x02, 0x03, 0x84, 0xFF, 0xFF, 0xFF, 0xFF])
    );
    assert_eq!(
        load_sign_extend_write_data(LOADW, read_data, 4),
        rv64_bytes_to_u16_block([0xAA, 0xBB, 0xCC, 0xDD, 0xFF, 0xFF, 0xFF, 0xFF])
    );
    // Misaligned within one block, negative.
    assert_eq!(
        load_sign_extend_write_data(LOADW, read_data, 3),
        rv64_bytes_to_u16_block([0x84, 0xAA, 0xBB, 0xCC, 0xFF, 0xFF, 0xFF, 0xFF])
    );
    // Misaligned across the block boundary, negative sign byte from the second block.
    assert_eq!(
        load_sign_extend_write_data(LOADW, read_data, 5),
        rv64_bytes_to_u16_block([0xBB, 0xCC, 0xDD, 156, 0xFF, 0xFF, 0xFF, 0xFF])
    );

    let read_data = [
        rv64_bytes_to_u16_block([0x01, 0x02, 0x03, 0x04, 0xAA, 0xBB, 0xCC, 0x7D]),
        block1,
    ];
    assert_eq!(
        load_sign_extend_write_data(LOADW, read_data, 0),
        rv64_bytes_to_u16_block([0x01, 0x02, 0x03, 0x04, 0, 0, 0, 0])
    );
    assert_eq!(
        load_sign_extend_write_data(LOADW, read_data, 4),
        rv64_bytes_to_u16_block([0xAA, 0xBB, 0xCC, 0x7D, 0, 0, 0, 0])
    );
    // Misaligned across the block boundary, positive sign byte from the second block.
    assert_eq!(
        load_sign_extend_write_data(LOADW, read_data, 6),
        rv64_bytes_to_u16_block([0xCC, 0x7D, 156, 92, 0, 0, 0, 0])
    );
}
