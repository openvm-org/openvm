pub mod fp12;
pub mod sw;

// Babybear
pub const FIELD_ELEMENT_BITS: usize = 30;

use num_bigint_dig::BigUint;

pub struct EcPoint {
    pub x: BigUint,
    pub y: BigUint,
}

pub(crate) struct FpBigUint(BigUint);

pub(crate) struct Fp2BigUint {
    pub c0: FpBigUint,
    pub c1: FpBigUint,
}

/// Fp12 represented as 6 Fp2 elements (each represented as 2 BigUints)
pub(crate) struct Fp12BigUint {
    pub c0: Fp2BigUint,
    pub c1: Fp2BigUint,
    pub c2: Fp2BigUint,
    pub c3: Fp2BigUint,
    pub c4: Fp2BigUint,
    pub c5: Fp2BigUint,
    // pub c00: BigUint,
    // pub c01: BigUint,
    // pub c10: BigUint,
    // pub c11: BigUint,
    // pub c20: BigUint,
    // pub c21: BigUint,
    // pub c30: BigUint,
    // pub c31: BigUint,
    // pub c40: BigUint,
    // pub c41: BigUint,
    // pub c50: BigUint,
    // pub c51: BigUint,
}
