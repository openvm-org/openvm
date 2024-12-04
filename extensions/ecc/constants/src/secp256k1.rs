use hex_literal::hex;
use lazy_static::lazy_static;
use num_bigint_dig::BigUint;

use super::CurveConst;

lazy_static! {
    pub static ref SECP256K1: CurveConst = CurveConst {
        MODULUS: BigUint::from_bytes_be(&hex!(
            "FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F"
        )),
        ORDER: BigUint::from_bytes_be(&hex!(
            "FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
        )),
        XI: [1, 1],  // doesn't apply to secp256k1
        NUM_LIMBS: 32,
        LIMB_BITS: 8,
        BLOCK_SIZE: 32,
    };
}
