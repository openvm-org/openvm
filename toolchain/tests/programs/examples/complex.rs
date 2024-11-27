#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm_algebra::{
    field::{Complex, ComplexConjugate},
    DivAssignUnsafe, DivUnsafe, IntMod,
};
use axvm_ecc::sw::{setup_fp2, Secp256k1Coord};

axvm::entry!(main);

pub fn main() {
    setup_fp2();
    let mut a = Complex::new(
        Secp256k1Coord::from_repr(core::array::from_fn(|_| 10)),
        Secp256k1Coord::from_repr(core::array::from_fn(|_| 21)),
    );
    let mut b = Complex::new(
        Secp256k1Coord::from_repr(core::array::from_fn(|_| 32)),
        Secp256k1Coord::from_repr(core::array::from_fn(|_| 47)),
    );

    for _ in 0..32 {
        let mut res = &a * &b;
        res += &a * &Complex::new(Secp256k1Coord::ZERO, -b.c1.double());
        res.div_assign_unsafe(&b * &b.clone().conjugate());

        if a.clone().div_unsafe(&b) - res != Complex::<Secp256k1Coord>::ZERO {
            panic!();
        }

        a *= &b;
        b *= &a;
    }

    if a == b {
        panic!();
    }
}
