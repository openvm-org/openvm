#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use openvm_algebra_guest::IntMod;

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_declare! {
    Mod1 { modulus = "998244353" },
    Mod2 { modulus = "1000000007" },
    Mod3 { modulus = "1000000009" },
    Mod4 { modulus = "987898789" },
}

openvm_algebra_complex_macros::complex_declare! {
    Complex2 { mod_type = Mod3 },
}

openvm::init!("openvm_init_complex_redundant_modulus.rs");

pub fn main() {
    let b = Complex2::new(Mod3::ZERO, Mod3::from_u32(1000000008));
    assert_eq!(b.clone() * &b * &b * &b * &b, b);
}
