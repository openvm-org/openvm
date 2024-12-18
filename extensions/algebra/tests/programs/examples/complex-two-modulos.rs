#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_algebra_guest::{DivUnsafe, IntMod};

openvm::entry!(main);

openvm_algebra_moduli_setup::moduli_declare! {
    Mod1 { modulus = "998244353" },
    Mod2 { modulus = "1000000007" }
}
openvm_algebra_moduli_setup::moduli_init! {
    "998244353", "1000000007"
}

openvm_algebra_complex_macros::complex_declare! {
    Complex1 { mod_type = Mod1 },
    Complex2 { mod_type = Mod2 },
}

openvm_algebra_complex_macros::complex_init! {
    Complex2 { mod_idx = 1 }, Complex1 { mod_idx = 0 },
}

pub fn main() {
    setup_all_complex_extensions();
    let a = Complex1::new(Mod1::ZERO, Mod1::from_u32(998244352));
    let b = Complex2::new(Mod2::ZERO, Mod2::from_u32(1000000006));
    assert_eq!(a.clone() * &a * &a * &a * &a, a);
    assert_eq!(b.clone() * &b * &b * &b * &b, b);
}
