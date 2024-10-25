pub mod pairing;
pub mod sw;

// Babybear
pub const FIELD_ELEMENT_BITS: usize = 30;

use num_bigint_dig::BigUint;

pub struct EcPoint {
    pub x: BigUint,
    pub y: BigUint,
}
