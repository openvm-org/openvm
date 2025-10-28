use openvm_sha2_air::{Sha256Config, Sha2BlockHasherSubairConfig, Sha384Config, Sha512Config};

use crate::{Sha2BlockHasherVmDigestColsRef, Sha2BlockHasherVmRoundColsRef};

pub trait Sha2BlockHasherVmConfig: Sha2BlockHasherSubairConfig {
    /// Width of the Sha2VmRoundCols
    const BLOCK_HASHER_ROUND_WIDTH: usize;
    /// Width of the Sha2DigestCols
    const BLOCK_HASHER_DIGEST_WIDTH: usize;
    /// Width of the Sha2BlockHasherCols
    const BLOCK_HASHER_WIDTH: usize;
}

impl Sha2BlockHasherVmConfig for Sha256Config {
    const BLOCK_HASHER_ROUND_WIDTH: usize =
        Sha2BlockHasherVmRoundColsRef::<u8>::width::<Sha256Config>();
    const BLOCK_HASHER_DIGEST_WIDTH: usize =
        Sha2BlockHasherVmDigestColsRef::<u8>::width::<Sha256Config>();
    const BLOCK_HASHER_WIDTH: usize =
        if Self::BLOCK_HASHER_ROUND_WIDTH > Self::BLOCK_HASHER_DIGEST_WIDTH {
            Self::BLOCK_HASHER_ROUND_WIDTH
        } else {
            Self::BLOCK_HASHER_DIGEST_WIDTH
        };
}

impl Sha2BlockHasherVmConfig for Sha512Config {
    const BLOCK_HASHER_ROUND_WIDTH: usize =
        Sha2BlockHasherVmRoundColsRef::<u8>::width::<Sha512Config>();
    const BLOCK_HASHER_DIGEST_WIDTH: usize =
        Sha2BlockHasherVmDigestColsRef::<u8>::width::<Sha512Config>();
    const BLOCK_HASHER_WIDTH: usize = if <Self as Sha2BlockHasherVmConfig>::BLOCK_HASHER_ROUND_WIDTH
        > Self::BLOCK_HASHER_DIGEST_WIDTH
    {
        Self::BLOCK_HASHER_ROUND_WIDTH
    } else {
        Self::BLOCK_HASHER_DIGEST_WIDTH
    };
}

impl Sha2BlockHasherVmConfig for Sha384Config {
    const BLOCK_HASHER_ROUND_WIDTH: usize =
        Sha2BlockHasherVmRoundColsRef::<u8>::width::<Sha384Config>();
    const BLOCK_HASHER_DIGEST_WIDTH: usize =
        Sha2BlockHasherVmDigestColsRef::<u8>::width::<Sha384Config>();
    const BLOCK_HASHER_WIDTH: usize =
        if Self::BLOCK_HASHER_ROUND_WIDTH > Self::BLOCK_HASHER_DIGEST_WIDTH {
            Self::BLOCK_HASHER_ROUND_WIDTH
        } else {
            Self::BLOCK_HASHER_DIGEST_WIDTH
        };
}
