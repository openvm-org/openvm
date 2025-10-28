use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_sha2_air::{Sha256Config, Sha2Config, Sha384Config, Sha512Config};
use openvm_sha2_transpiler::Rv32Sha2Opcode;

use super::{
    Sha2BlockHasherControlColsRef, Sha2BlockHasherDigestColsRef, Sha2BlockHasherRoundColsRef,
};

pub trait Sha2ChipConfig: Sha2BlockHasherConfig {
    // Name of the opcode
    const OPCODE_NAME: &'static str;
    /// Width of the ShaVmControlCols
    const BLOCK_HASHER_CONTROL_WIDTH: usize = Sha2BlockHasherControlColsRef::<u8>::width::<Self>();
    /// Width of the ShaVmRoundCols
    const VM_ROUND_WIDTH: usize = Sha2BlockHasherRoundColsRef::<u8>::width::<Self>();
    /// Width of the ShaVmDigestCols
    const VM_DIGEST_WIDTH: usize = Sha2BlockHasherDigestColsRef::<u8>::width::<Self>();
    /// Width of the ShaVmCols
    const VM_WIDTH: usize = if Self::VM_ROUND_WIDTH > Self::VM_DIGEST_WIDTH {
        Self::VM_ROUND_WIDTH
    } else {
        Self::VM_DIGEST_WIDTH
    };
    /// Number of bits to use when padding the message length. Given by the SHA-2 spec.
    const MESSAGE_LENGTH_BITS: usize;
    /// Maximum i such that `FirstPadding_i` is a valid padding flag
    const MAX_FIRST_PADDING: usize = Self::CELLS_PER_ROW - 1;
    /// Maximum i such that `FirstPadding_i_LastRow` is a valid padding flag
    const MAX_FIRST_PADDING_LAST_ROW: usize =
        Self::CELLS_PER_ROW - Self::MESSAGE_LENGTH_BITS / 8 - 1;
    /// OpenVM Opcode for the instruction
    const OPCODE: Rv32Sha2Opcode;

    // ==== Constants for register/memory adapter ====
    /// Number of rv32 cells read in a block
    const BLOCK_CELLS: usize = Self::BLOCK_BITS / RV32_CELL_BITS;
    /// Number of rows we will do a read on for each block
    const NUM_READ_ROWS: usize = Self::MESSAGE_ROWS;

    /// Number of cells to read in a single memory access
    const READ_SIZE: usize = Self::WORD_U8S * Self::ROUNDS_PER_ROW;
    /// Number of cells in the digest before truncation (Sha384 truncates the digest)
    const HASH_SIZE: usize = Self::WORD_U8S * Self::HASH_WORDS;
    /// Number of cells in the digest after truncation
    const DIGEST_SIZE: usize;

    /// Number of parts to write the hash in
    const NUM_WRITES: usize = Self::HASH_SIZE / Self::WRITE_SIZE;
    /// Size of each write. Must divide Self::HASH_SIZE
    const WRITE_SIZE: usize;
}

/// Register reads to get dst, src, len
pub const SHA_REGISTER_READS: usize = 3;

impl Sha2ChipConfig for Sha256Config {
    const OPCODE_NAME: &'static str = "SHA256";
    const MESSAGE_LENGTH_BITS: usize = 64;
    const WRITE_SIZE: usize = SHA_WRITE_SIZE;
    const OPCODE: Rv32Sha2Opcode = Rv32Sha2Opcode::SHA256;
    // no truncation
    const DIGEST_SIZE: usize = Self::HASH_SIZE;
}

impl Sha2ChipConfig for Sha512Config {
    const OPCODE_NAME: &'static str = "SHA512";
    const MESSAGE_LENGTH_BITS: usize = 128;
    const WRITE_SIZE: usize = SHA_WRITE_SIZE;
    const OPCODE: Rv32Sha2Opcode = Rv32Sha2Opcode::SHA512;
    // no truncation
    const DIGEST_SIZE: usize = Self::HASH_SIZE;
}

impl Sha2ChipConfig for Sha384Config {
    const OPCODE_NAME: &'static str = "SHA384";
    const MESSAGE_LENGTH_BITS: usize = 128;
    const WRITE_SIZE: usize = SHA_WRITE_SIZE;
    const OPCODE: Rv32Sha2Opcode = Rv32Sha2Opcode::SHA384;
    // Sha384 truncates the output to 48 cells
    const DIGEST_SIZE: usize = 48;
}

// We use the same write size for all variants to simplify tracegen record storage.
// In particular, each memory write aux record will have the same size, which is useful for
// defining Sha2VmRecordHeader in a repr(C) way.
pub const SHA_WRITE_SIZE: usize = 32;

pub const MAX_SHA_NUM_WRITES: usize = if Sha256Config::NUM_WRITES > Sha512Config::NUM_WRITES {
    if Sha256Config::NUM_WRITES > Sha384Config::NUM_WRITES {
        Sha256Config::NUM_WRITES
    } else {
        Sha384Config::NUM_WRITES
    }
} else if Sha512Config::NUM_WRITES > Sha384Config::NUM_WRITES {
    Sha512Config::NUM_WRITES
} else {
    Sha384Config::NUM_WRITES
};

/// Maximum message length that this chip supports in bytes
pub const SHA_MAX_MESSAGE_LEN: usize = 1 << 29;
