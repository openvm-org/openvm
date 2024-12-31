#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_algebra_guest::{moduli_setup::*, DivUnsafe, IntMod};

openvm::entry!(main);

// This macro will create two structs, `Mod1` and `Mod2`,
// one for arithmetic modulo 998244353, and the other for arithmetic modulo 1000000007.
moduli_declare! {
    Mod1 { modulus = "998244353" },
    Mod2 { modulus = "1000000007" }
}

// This macro will initialize the moduli.
// Now, `Mod1` is the "zeroth" modular struct, and `Mod2` is the "first" one.
moduli_init! {
    "998244353", "1000000007"
}

// This macro will create two structs, `Complex1` and `Complex2`,
// one for arithmetic in the field $\mathbb{F}_{998244353}[x]/(x^2 + 1)$,
// and the other for arithmetic in the field $\mathbb{F}_{1000000007}[x]/(x^2 + 1)$.
openvm_algebra_complex_macros::complex_declare! {
    Complex1 { mod_type = Mod1 },
    Complex2 { mod_type = Mod2 },
}

// The order of these structs does not matter,
// given that we specify the `mod_idx` parameters properly.
openvm_algebra_complex_macros::complex_init! {
    Complex2 { mod_idx = 1 }, Complex1 { mod_idx = 0 },
}

pub fn main() {
    // Since we only use an arithmetic operation with `Mod1` and not `Mod2`,
    // we only need to call `setup_0()` here.
    setup_0();
    setup_all_complex_extensions();
    let a = Complex1::new(Mod1::ZERO, Mod1::from_u32(0x3b8) * Mod1::from_u32(0x100000)); // a = -i in the corresponding field
    let b = Complex2::new(Mod2::ZERO, Mod2::from_u32(1000000006)); // b = -i in the corresponding field
    assert_eq!(a.clone() * &a * &a * &a * &a, a); // a^5 = a
    assert_eq!(b.clone() * &b * &b * &b * &b, b); // b^5 = b
                                                  // Note that these assertions would fail, have we provided the `mod_idx` parameters wrongly.
}
