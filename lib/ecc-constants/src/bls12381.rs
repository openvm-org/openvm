use lazy_static::lazy_static;
use num_bigint_dig::BigUint;
use num_traits::Num;

use super::CurveConst;

lazy_static! {
    pub static ref BLS12381: CurveConst = CurveConst {
        MODULUS: BigUint::from_str_radix(
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
            16,
        )
        .unwrap(),
        ORDER: BigUint::from_str_radix(
            "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
            16,
        )
        .unwrap(),
        XI: [1, 1],
    };
}
