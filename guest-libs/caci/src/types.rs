//! Rust mirrors of the C types declared in the standard's `zkvm_accelerators.h`.

/// Status code returned by every accelerator function (`zkvm_status`).
///
/// C declares this as an enum with a negative member, which the C ABI represents as `int`.
#[repr(i32)]
pub enum ZkvmStatus {
    /// `ZKVM_EOK`: success.
    Ok = 0,
    /// `ZKVM_EFAIL`: failure.
    Fail = -1,
}

#[repr(C, align(8))]
pub struct ZkvmBytes16 {
    pub data: [u8; 16],
}

#[repr(C, align(8))]
pub struct ZkvmBytes32 {
    pub data: [u8; 32],
}

#[repr(C, align(8))]
pub struct ZkvmBytes48 {
    pub data: [u8; 48],
}

#[repr(C, align(8))]
pub struct ZkvmBytes64 {
    pub data: [u8; 64],
}

#[repr(C, align(8))]
pub struct ZkvmBytes96 {
    pub data: [u8; 96],
}

#[repr(C, align(8))]
pub struct ZkvmBytes128 {
    pub data: [u8; 128],
}

#[repr(C, align(8))]
pub struct ZkvmBytes192 {
    pub data: [u8; 192],
}

/// Hash types
pub type ZkvmKeccak256Hash = ZkvmBytes32;
pub type ZkvmSha256Hash = ZkvmBytes32;
pub type ZkvmRipemd160Hash = ZkvmBytes32; // 20-byte hash padded to 32 bytes, first 12 bytes are zero.

/// secp256k1 types
pub type ZkvmSecp256k1Hash = ZkvmBytes32;
pub type ZkvmSecp256k1Signature = ZkvmBytes64;
pub type ZkvmSecp256k1Pubkey = ZkvmBytes64;

/// secp256r1 (P-256) types
pub type ZkvmSecp256r1Hash = ZkvmBytes32;
pub type ZkvmSecp256r1Signature = ZkvmBytes64;
pub type ZkvmSecp256r1Pubkey = ZkvmBytes64;

/// BN254 types
pub type ZkvmBn254G1Point = ZkvmBytes64;
pub type ZkvmBn254G2Point = ZkvmBytes128;
pub type ZkvmBn254Scalar = ZkvmBytes32;

#[repr(C)]
pub struct ZkvmBn254PairingPair {
    pub g1: ZkvmBn254G1Point,
    pub g2: ZkvmBn254G2Point,
}

/// BLS12-381 types
pub type ZkvmBls12_381G1Point = ZkvmBytes96;
pub type ZkvmBls12_381G2Point = ZkvmBytes192;
pub type ZkvmBls12_381Scalar = ZkvmBytes32;

pub type ZkvmBls12_381Fp = ZkvmBytes48;
pub type ZkvmBls12_381Fp2 = ZkvmBytes96;

#[repr(C)]
pub struct ZkvmBls12_381G1MsmPair {
    pub point: ZkvmBls12_381G1Point,
    pub scalar: ZkvmBls12_381Scalar,
}

#[repr(C)]
pub struct ZkvmBls12_381G2MsmPair {
    pub point: ZkvmBls12_381G2Point,
    pub scalar: ZkvmBls12_381Scalar,
}

#[repr(C)]
pub struct ZkvmBls12_381PairingPair {
    pub g1: ZkvmBls12_381G1Point,
    pub g2: ZkvmBls12_381G2Point,
}

/// BLAKE2f types
pub type ZkvmBlake2fState = ZkvmBytes64;
pub type ZkvmBlake2fMessage = ZkvmBytes128;
pub type ZkvmBlake2fOffset = ZkvmBytes16;

/// KZG types
pub type ZkvmKzgCommitment = ZkvmBytes48;
pub type ZkvmKzgProof = ZkvmBytes48;
pub type ZkvmKzgFieldElement = ZkvmBytes32;
