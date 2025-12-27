#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use serde::{Deserialize, Serialize};
use serde_with::{serde_as, Bytes};

/// Signature recovery test vectors
#[repr(C)]
#[serde_as]
#[derive(Serialize, Deserialize)]
pub struct RecoveryTestVector {
    #[serde_as(as = "Bytes")]
    pub pk: [u8; 33],
    #[serde_as(as = "Bytes")]
    pub msg: [u8; 32],
    #[serde_as(as = "Bytes")]
    pub sig: [u8; 64],
    pub recid: u8,
    pub ok: bool,
}

#[repr(C)]
#[serde_as]
#[derive(Serialize, Deserialize)]
pub struct Sec1DecodingTestVector {
    #[serde_as(as = "Bytes")]
    pub bytes: Vec<u8>,
    pub ok: bool,
}
