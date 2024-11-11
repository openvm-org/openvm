#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm::intrinsics::IntMod;
use axvm_ecc::{field::Complex, sw::IntModN};

axvm::entry!(main);

pub fn main() {
    let mut a = Complex::new(
        IntModN::from_repr(core::array::from_fn(|_| 10)),
        IntModN::from_repr(core::array::from_fn(|_| 21)),
    );
    let mut b = Complex::new(
        IntModN::from_repr(core::array::from_fn(|_| 32)),
        IntModN::from_repr(core::array::from_fn(|_| 47)),
    );

    for _ in 0..32 {
        let mut res = &a * &b;
        res += &a * &Complex::new(IntModN::ZERO, -b.c1.double());
        res /= &b * &b.conjugate();

        if (&a / &b) - res != Complex::<IntModN>::ZERO {
            panic!();
        }

        a *= &b;
        b *= &a;
    }

    if a == b {
        panic!();
    }
}
