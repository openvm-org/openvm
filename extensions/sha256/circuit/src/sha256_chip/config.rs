use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_sha_air::{Sha256Config, Sha512Config, ShaConfig};

use super::{ShaVmControlColsRef, ShaVmDigestColsRef, ShaVmRoundColsRef};

pub trait ShaChipConfig: ShaConfig {
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

    // ==== Constants for register/memory adapter ====
    // Number of rv32 cells read in a SHA256 block
    const BLOCK_CELLS: usize = Self::BLOCK_BITS / RV32_CELL_BITS;
    /// Number of rows we will do a read on for each SHA256 block
    const NUM_READ_ROWS: usize = Self::BLOCK_CELLS / SHA_READ_SIZE;
}

/// Number of cells to read in a single memory access
pub const SHA_READ_SIZE: usize = 16;
/// Number of cells to write in a single memory access.
pub const SHA_WRITE_SIZE: usize = 32;

/// Register reads to get dst, src, len
pub const SHA_REGISTER_READS: usize = 3;

impl ShaChipConfig for Sha256Config {
    const OPCODE_NAME: &'static str = "SHA256";
}

// Currently same as Sha256Config, but can configure later
impl ShaChipConfig for Sha512Config {
    const OPCODE_NAME: &'static str = "SHA512";
}
