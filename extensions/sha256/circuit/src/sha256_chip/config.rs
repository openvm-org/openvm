use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_sha256_transpiler::Rv32Sha2Opcode;
use openvm_sha_air::{Sha256Config, Sha384Config, Sha512Config, ShaConfig};

use super::{ShaVmControlColsRef, ShaVmDigestColsRef, ShaVmRoundColsRef};

pub enum Sha2Variant {
    Sha256,
    Sha512,
    Sha384,
}

pub trait ShaChipConfig: ShaConfig {
    // Differentiate between the two SHA256 variants
    const VARIANT: Sha2Variant;
    // Name of the opcode
    const OPCODE_NAME: &'static str;
    /// Width of the ShaVmControlCols
    const VM_CONTROL_WIDTH: usize = ShaVmControlColsRef::<u8>::width::<Self>();
    /// Width of the ShaVmRoundCols
    const VM_ROUND_WIDTH: usize = ShaVmRoundColsRef::<u8>::width::<Self>();
    /// Width of the ShaVmDigestCols
    const VM_DIGEST_WIDTH: usize = ShaVmDigestColsRef::<u8>::width::<Self>();
    /// Width of the ShaVmCols
    const VM_WIDTH: usize = if Self::VM_ROUND_WIDTH > Self::VM_DIGEST_WIDTH {
        Self::VM_ROUND_WIDTH
    } else {
        Self::VM_DIGEST_WIDTH
    };
    /// Number of bits to use when padding the message length
    const MESSAGE_LENGTH_BITS: usize;
    /// Maximum i such that `FirstPadding_i` is a valid padding flag
    const MAX_FIRST_PADDING: usize = Self::CELLS_PER_ROW - 1;
    /// Maximum i such that `FirstPadding_i_LastRow` is a valid padding flag
    const MAX_FIRST_PADDING_LAST_ROW: usize =
        Self::CELLS_PER_ROW - Self::MESSAGE_LENGTH_BITS / 8 - 1;
    /// OpenVM Opcode for the instruction
    const OPCODE: Rv32Sha2Opcode;

    // ==== Constants for register/memory adapter ====
    // Number of rv32 cells read in a SHA256 block
    const BLOCK_CELLS: usize = Self::BLOCK_BITS / RV32_CELL_BITS;
    /// Number of rows we will do a read on for each SHA256 block
    const NUM_READ_ROWS: usize = Self::MESSAGE_ROWS;

    /// Number of cells to read in a single memory access
    const READ_SIZE: usize = Self::WORD_U8S * Self::ROUNDS_PER_ROW;
    /// Number of cells in the digest before truncation (Sha384 truncates the digest)
    const HASH_SIZE: usize = Self::WORD_U8S * Self::HASH_WORDS;
    /// Number of cells in the digest after truncation
    const DIGEST_SIZE: usize;

    /// Number of parts to write the hash in. Must divide HASH_SIZE
    const NUM_WRITES: usize;
    /// Size of each write
    const WRITE_SIZE: usize = Self::HASH_SIZE / Self::NUM_WRITES;
}

/// Register reads to get dst, src, len
pub const SHA_REGISTER_READS: usize = 3;

impl ShaChipConfig for Sha256Config {
    const VARIANT: Sha2Variant = Sha2Variant::Sha256;
    const OPCODE_NAME: &'static str = "SHA256";
    const MESSAGE_LENGTH_BITS: usize = 64;
    const NUM_WRITES: usize = 1;
    const OPCODE: Rv32Sha2Opcode = Rv32Sha2Opcode::SHA256;
    // no truncation
    const DIGEST_SIZE: usize = Self::HASH_SIZE;
}

// Currently same as Sha256Config, but can configure later
impl ShaChipConfig for Sha512Config {
    const VARIANT: Sha2Variant = Sha2Variant::Sha512;
    const OPCODE_NAME: &'static str = "SHA512";
    const MESSAGE_LENGTH_BITS: usize = 128;
    // Use 2 writes because we only support writes up to 32 bytes
    const NUM_WRITES: usize = 2;
    const OPCODE: Rv32Sha2Opcode = Rv32Sha2Opcode::SHA512;
    // no truncation
    const DIGEST_SIZE: usize = Self::HASH_SIZE;
}

impl ShaChipConfig for Sha384Config {
    const VARIANT: Sha2Variant = Sha2Variant::Sha384;
    const OPCODE_NAME: &'static str = "SHA384";
    const MESSAGE_LENGTH_BITS: usize = 128;
    // Use 2 writes because we only support writes up to 32 bytes
    const NUM_WRITES: usize = 2;
    const OPCODE: Rv32Sha2Opcode = Rv32Sha2Opcode::SHA384;
    // Sha284 truncates the output to 48 cells
    const DIGEST_SIZE: usize = 48;
}
