#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_algebra_guest::IntMod;

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_declare! {
    Mod1 { modulus = "998244353" },
    Mod2 { modulus = "1000000007" }
}

// the order of the moduli here does not match the order in the config
openvm_algebra_moduli_macros::moduli_init! {
    "1000000007",
    "998244353",
}

pub fn main() {
    // this should cause a debug assertion to fail
    setup_all_moduli();
}
