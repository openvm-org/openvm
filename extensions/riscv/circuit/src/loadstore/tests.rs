use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{
    LOADB, LOADBU, LOADD, LOADH, LOADHU, LOADW, LOADWU, STOREB, STOREH,
};

use super::test_utils::*;

#[test]
fn load_sign_extend_sanity_tests() {
    let read_data = b([34, 159, 237, 151, 100, 200, 50, 25]);
    assert_eq!(
        run_write_data(LOADH, read_data, [0; BLOCK_FE_WIDTH], 0),
        b([34, 159, 255, 255, 255, 255, 255, 255])
    );
    assert_eq!(
        run_write_data(LOADH, read_data, [0; BLOCK_FE_WIDTH], 2),
        b([237, 151, 255, 255, 255, 255, 255, 255])
    );
    assert_eq!(
        run_write_data(LOADH, read_data, [0; BLOCK_FE_WIDTH], 4),
        b([100, 200, 255, 255, 255, 255, 255, 255])
    );
    assert_eq!(
        run_write_data(LOADH, read_data, [0; BLOCK_FE_WIDTH], 6),
        b([50, 25, 0, 0, 0, 0, 0, 0])
    );

    let read_data = b([45, 82, 99, 127, 200, 150, 180, 210]);
    for shift in 0..8 {
        let byte = rv64_u16_block_to_bytes(read_data)[shift];
        assert_eq!(
            rv64_u16_block_to_bytes(run_write_data(LOADB, read_data, [0; BLOCK_FE_WIDTH], shift)),
            (byte as i8 as i64).to_le_bytes(),
            "LOADB shift={shift}"
        );
    }

    let read_data = b([0x01, 0x02, 0x03, 0x84, 0xAA, 0xBB, 0xCC, 0xDD]);
    assert_eq!(
        run_write_data(LOADW, read_data, [0; BLOCK_FE_WIDTH], 0),
        b([0x01, 0x02, 0x03, 0x84, 0xFF, 0xFF, 0xFF, 0xFF])
    );
    assert_eq!(
        run_write_data(LOADW, read_data, [0; BLOCK_FE_WIDTH], 4),
        b([0xAA, 0xBB, 0xCC, 0xDD, 0xFF, 0xFF, 0xFF, 0xFF])
    );

    let read_data = b([0x01, 0x02, 0x03, 0x04, 0xAA, 0xBB, 0xCC, 0x7D]);
    assert_eq!(
        run_write_data(LOADW, read_data, [0; BLOCK_FE_WIDTH], 0),
        b([0x01, 0x02, 0x03, 0x04, 0, 0, 0, 0])
    );
    assert_eq!(
        run_write_data(LOADW, read_data, [0; BLOCK_FE_WIDTH], 4),
        b([0xAA, 0xBB, 0xCC, 0x7D, 0, 0, 0, 0])
    );
}

#[test]
fn negative_split_write_data_tests() {
    assert_pranked_byte_fails(STOREB, |core| core.read_data[0] += F::ONE);
    assert_pranked_halfword_fails(LOADHU, |core| core.read_data[0] += F::ONE);
    assert_pranked_word_fails(LOADWU, |core| core.read_data[0] += F::ONE);
    assert_pranked_doubleword_fails(LOADD, |core| core.read_data[0] += F::ONE);
}

#[test]
fn negative_split_opcode_role_tests() {
    assert_pranked_byte_fails(LOADBU, |core| core.is_load = F::ZERO);
    assert_pranked_halfword_fails(STOREH, |core| core.is_load = F::ONE);
    assert_pranked_word_fails(LOADWU, |core| core.is_load = F::ZERO);
    assert_pranked_doubleword_fails(LOADD, |core| core.is_load = F::ZERO);
}
