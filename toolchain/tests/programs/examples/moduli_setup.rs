#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use axvm_algebra::IntMod;

axvm::entry!(main);
axvm::moduli_setup! {
    bls12381 { modulus = "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787" },
    Mod1e18 { modulus = "1000000000000000003" },
}

axvm::moduli_setup! {
    Mersenne61 { modulus = "0x1fffffffffffffff" },
}

pub fn main() {
    setup_bls12381();
    setup_Mod1e18();
    setup_Mersenne61();
    let x = bls12381::from_repr(core::array::from_fn(|i| i as u8));
    assert_eq!(x.0.len(), 48);

    let y = Mod1e18::from_u32(100);
    let y = (&y * &y) * &y;
    let y = y.clone() * y.clone() * y.clone();
    assert_eq!(y + Mod1e18::from_u32(3), Mod1e18::ZERO);

    let mut res = Mersenne61::from_u32(1);
    for _ in 0..61 {
        res += res.clone();
    }
    assert_eq!(res, Mersenne61::from_u32(1));
}
