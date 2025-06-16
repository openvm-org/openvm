#![no_std]

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
