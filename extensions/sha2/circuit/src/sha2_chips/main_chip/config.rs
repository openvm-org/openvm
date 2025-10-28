use openvm_sha2_transpiler::Rv32Sha2Opcode;

use crate::{
    Sha256Config, Sha2ColsRef, Sha384Config, Sha512Config, SHA2_READ_SIZE, SHA2_REGISTER_READS,
    SHA2_WRITE_SIZE,
};

pub trait Sha2MainChipConfig: Send + Sync + Clone {
    // --- Required ---
    /// Number of bytes in a SHA block
    const BLOCK_BYTES: usize;
    /// Number of bytes in a SHA state
    const STATE_BYTES: usize;
    /// Number of bytes in a SHA digest
    const DIGEST_BYTES: usize;
    /// OpenVM Opcode for the instruction
    const OPCODE: Rv32Sha2Opcode;

    // --- Provided ---
    const BLOCK_READS: usize = Self::BLOCK_BYTES / SHA2_READ_SIZE;
    const STATE_READS: usize = Self::STATE_BYTES / SHA2_READ_SIZE;
    const DIGEST_WRITES: usize = Self::DIGEST_BYTES / SHA2_WRITE_SIZE;

    const TIMESTAMP_DELTA: usize =
        Self::BLOCK_READS + Self::STATE_READS + Self::DIGEST_WRITES + SHA2_REGISTER_READS;

    const WIDTH: usize = Sha2ColsRef::<u8>::width::<Self>();
}

impl Sha2MainChipConfig for Sha256Config {
    const BLOCK_BYTES: usize = 64;
    const STATE_BYTES: usize = 32;
    const DIGEST_BYTES: usize = 32;
    const OPCODE: Rv32Sha2Opcode = Rv32Sha2Opcode::SHA256;
}

impl Sha2MainChipConfig for Sha512Config {
    const BLOCK_BYTES: usize = 128;
    const STATE_BYTES: usize = 64;
    const DIGEST_BYTES: usize = 64;
    const OPCODE: Rv32Sha2Opcode = Rv32Sha2Opcode::SHA512;
}

impl Sha2MainChipConfig for Sha384Config {
    const BLOCK_BYTES: usize = Sha512Config::BLOCK_BYTES;
    const STATE_BYTES: usize = Sha512Config::STATE_BYTES;
    const DIGEST_BYTES: usize = 48;
    const OPCODE: Rv32Sha2Opcode = Rv32Sha2Opcode::SHA384;
}
