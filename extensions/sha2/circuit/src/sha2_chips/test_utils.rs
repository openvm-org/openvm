use std::{
    array,
    borrow::BorrowMut,
    sync::{Arc, Mutex},
};

use hex::FromHex;
use itertools::Itertools;
use openvm_circuit::{
    arch::{
        testing::{
            memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder,
            BITWISE_OP_LOOKUP_BUS,
        },
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::{
        memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
        SystemPort,
    },
    utils::get_random_message,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS},
    LocalOpcode,
};
use openvm_sha2_air::{
    word_into_u8_limbs, Sha256Config, Sha2BlockHasherSubairConfig, Sha2Variant, Sha512Config,
};
use openvm_sha2_transpiler::Rv32Sha2Opcode::{self, *};
use openvm_stark_backend::{
    interaction::BusIndex,
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
#[cfg(feature = "cuda")]
use {
    crate::{trace::Sha2BlockHasherRecordMut, Sha2BlockHasherChipGpu},
    openvm_circuit::arch::testing::{
        default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness,
    },
};

use crate::{
    Sha2BlockHasherChip, Sha2BlockHasherVmAir, Sha2Config, Sha2MainAir, Sha2MainChip,
    Sha2MainChipConfig, Sha2VmExecutor, SHA2_READ_SIZE, SHA2_WRITE_SIZE,
};

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
