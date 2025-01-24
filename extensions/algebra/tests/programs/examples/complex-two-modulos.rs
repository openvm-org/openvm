#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_algebra_guest::IntMod;

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_declare! {
    Mod1 { modulus = "998244353" },
    Mod2 { modulus = "1000000007" },
    Mod3 { modulus = "1000000009" },
    Mod4 { modulus = "987898789" },
}
openvm_algebra_moduli_macros::moduli_init! {
    "998244353", "1000000007", "1000000009", "987898789"
}

openvm_algebra_complex_macros::complex_declare! {
    Complex2 { mod_type = Mod3 },
}

openvm_algebra_complex_macros::complex_init! {
    Complex2 { mod_idx = 2 },
}

pub fn main() {
    setup_all_moduli();
    setup_all_complex_extensions();
    let b = Complex2::new(Mod3::ZERO, Mod3::from_u32(1000000008));
    assert_eq!(b.clone() * &b * &b * &b * &b, b);
}
