use itertools::Itertools;
use openvm_circuit::arch::testing::TestBuilder;
use openvm_instructions::riscv::RV32_MEMORY_AS;
use openvm_sha2_air::Sha2Variant;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{Sha2Config, SHA2_READ_SIZE, SHA2_WRITE_SIZE};

// See https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf for the padding algorithm
pub fn add_padding_to_message<C: Sha2Config + 'static>(mut message: Vec<u8>) -> Vec<u8> {
    // Length of the message in bits
    let message_len = message.len() * 8;

    // For SHA-256,
    // l + 1 + k = 448 mod 512
    // <=> l + 1 + k + 8 = 0 mod 512
    // <=> k = -(l + 1 + 8) mod 512
    // <=> k = (512 - (l + 1 + 8)) mod 512
    // The other variants are similar.
    let padding_len_bits = (C::BLOCK_BITS
        - ((message_len + 1 + C::MESSAGE_LENGTH_BITS) % C::BLOCK_BITS))
        % C::BLOCK_BITS;
    message.push(0x80);

    let padding_len_bytes = padding_len_bits / 8;
    message.extend(std::iter::repeat_n(0x00, padding_len_bytes));

    match C::VARIANT {
        Sha2Variant::Sha256 => {
            message.extend_from_slice(&((message_len as u64).to_be_bytes()));
        }
        Sha2Variant::Sha512 => {
            message.extend_from_slice(&((message_len as u128).to_be_bytes()));
        }
        Sha2Variant::Sha384 => {
            message.extend_from_slice(&((message_len as u128).to_be_bytes()));
        }
    };

    assert_eq!(message.len() % C::BLOCK_BYTES, 0);

    message
}

pub fn write_slice_to_memory<F: PrimeField32>(
    tester: &mut impl TestBuilder<F>,
    data: &[u8],
    ptr: usize,
) {
    data.chunks_exact(4).enumerate().for_each(|(i, chunk)| {
        tester.write::<SHA2_WRITE_SIZE>(
            RV32_MEMORY_AS as usize,
            ptr + i * 4,
            chunk
                .iter()
                .cloned()
                .map(F::from_canonical_u8)
                .collect_vec()
                .try_into()
                .unwrap(),
        );
    });
}

pub fn read_slice_from_memory<F: PrimeField32>(
    tester: &mut impl TestBuilder<F>,
    ptr: usize,
    len: usize,
) -> Vec<F> {
    let mut data = Vec::new();
    for i in 0..(len / SHA2_READ_SIZE) {
        data.extend_from_slice(
            &tester.read::<SHA2_READ_SIZE>(RV32_MEMORY_AS as usize, ptr + i * SHA2_READ_SIZE),
        );
    }
    data
}
