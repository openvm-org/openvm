use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{LOADB, LOADH, LOADW};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;

use super::test_utils::{
    assert_pranked_byte_fails, assert_pranked_halfword_fails, assert_pranked_word_fails,
    load_sign_extend_write_data, rv64_u16_block_to_bytes, F,
};
use crate::test_utils::memory::rv64_bytes_to_u16_block;

#[test]
fn load_sign_extend_sanity_tests() {
    let read_data = rv64_bytes_to_u16_block([34, 159, 237, 151, 100, 200, 50, 25]);
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

    let read_data = rv64_bytes_to_u16_block([45, 82, 99, 127, 200, 150, 180, 210]);
    for shift in 0..8 {
        let byte = rv64_u16_block_to_bytes(read_data)[shift];
        assert_eq!(
            rv64_u16_block_to_bytes(load_sign_extend_write_data(LOADB, read_data, shift)),
            (byte as i8 as i64).to_le_bytes(),
            "LOADB shift={shift}"
        );
    }

    let read_data = rv64_bytes_to_u16_block([0x01, 0x02, 0x03, 0x84, 0xAA, 0xBB, 0xCC, 0xDD]);
    assert_eq!(
        load_sign_extend_write_data(LOADW, read_data, 0),
        rv64_bytes_to_u16_block([0x01, 0x02, 0x03, 0x84, 0xFF, 0xFF, 0xFF, 0xFF])
    );
    assert_eq!(
        load_sign_extend_write_data(LOADW, read_data, 4),
        rv64_bytes_to_u16_block([0xAA, 0xBB, 0xCC, 0xDD, 0xFF, 0xFF, 0xFF, 0xFF])
    );

    let read_data = rv64_bytes_to_u16_block([0x01, 0x02, 0x03, 0x04, 0xAA, 0xBB, 0xCC, 0x7D]);
    assert_eq!(
        load_sign_extend_write_data(LOADW, read_data, 0),
        rv64_bytes_to_u16_block([0x01, 0x02, 0x03, 0x04, 0, 0, 0, 0])
    );
    assert_eq!(
        load_sign_extend_write_data(LOADW, read_data, 4),
        rv64_bytes_to_u16_block([0xAA, 0xBB, 0xCC, 0x7D, 0, 0, 0, 0])
    );
}

#[test]
fn negative_split_signed_load_tests() {
    assert_pranked_byte_fails(|core| core.data_most_sig_bit += F::ONE);
    assert_pranked_halfword_fails(|core| core.data_most_sig_bit += F::ONE);
    assert_pranked_word_fails(|core| core.data_most_sig_bit += F::ONE);
}
