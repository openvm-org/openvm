use lazy_static::lazy_static;
use num_bigint_dig::BigUint;
use num_traits::Num;

use super::CurveConst;

lazy_static! {
    pub static ref BN254: CurveConst = CurveConst {
        MODULUS: BigUint::from_str_radix(
            "21888242871839275222246405745257275088696311157297823662689037894645226208583",
            10,
        )
        .unwrap(),
        ORDER: BigUint::from_str_radix(
            "21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10,
        )
        .unwrap(),
        XI: [9, 1],
    };
}
