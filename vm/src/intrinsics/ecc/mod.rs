pub mod fp12;
pub mod pairing;
pub mod sw;

use num_bigint_dig::BigUint;

pub struct EcPoint {
    pub x: BigUint,
    pub y: BigUint,
}

pub struct FpBigUint(pub BigUint);

pub struct Fp2BigUint {
    pub c0: FpBigUint,
    pub c1: FpBigUint,
}

/// Fp12 represented as 6 Fp2 elements (each represented as 2 BigUints)
pub struct Fp12BigUint {
    pub c0: Fp2BigUint,
    pub c1: Fp2BigUint,
    pub c2: Fp2BigUint,
    pub c3: Fp2BigUint,
    pub c4: Fp2BigUint,
    pub c5: Fp2BigUint,
}
