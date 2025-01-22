#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_algebra_guest::{moduli_setup::*, IntMod};

openvm::entry!(main);

// This macro will create `Mod1`
moduli_declare! {
    Mod1 { modulus = "875597534642899606784368692118018719973065508781036741540935024101872394795853278954847377898236" },
}

moduli_init! {
    "875597534642899606784368692118018719973065508781036741540935024101872394795853278954847377898236"
}

pub fn main() {
    setup_all_moduli();

    assert!(Mod1::ONE != Mod1::ZERO);
}
