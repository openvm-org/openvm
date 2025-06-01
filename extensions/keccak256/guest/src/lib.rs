#![no_std]

/// This is custom-0 defined in RISC-V spec document
pub const OPCODE: u8 = 0x0b;
pub const KECCAK256_FUNCT3: u8 = 0b100;
pub const KECCAK256_FUNCT7: u8 = 0;

/// Native hook for keccak256 for use with `alloy-primitives` "native-keccak" feature.
///
/// # Safety
///
/// The VM accepts the preimage by pointer and length, and writes the
/// 32-byte hash.
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 32-bytes long.
///
/// [`keccak256`]: https://en.wikipedia.org/wiki/SHA-3
/// [`sha3`]: https://docs.rs/sha3/latest/sha3/
/// [`tiny_keccak`]: https://docs.rs/tiny-keccak/latest/tiny_keccak/
#[cfg(target_os = "zkvm")]
#[inline(always)]
#[no_mangle]
pub extern "C" fn native_keccak256(bytes: *const u8, len: usize, output: *mut u8) {
    openvm_platform::custom_insn_r!(
        opcode = OPCODE,
        funct3 = KECCAK256_FUNCT3,
        funct7 = KECCAK256_FUNCT7,
        rd = In output,
        rs1 = In bytes,
        rs2 = In len
    );
}
