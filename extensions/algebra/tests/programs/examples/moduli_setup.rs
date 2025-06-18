#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use openvm_algebra_guest::IntMod;

openvm::entry!(main);
openvm_algebra_moduli_macros::moduli_declare! {
    Bls12381 { modulus = "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787" },
    Mod1e18 { modulus = "1000000000000000003" },
}

openvm_algebra_moduli_macros::moduli_declare! {
    Mersenne61 { modulus = "0x1fffffffffffffff" },
}

openvm::init!("openvm_init_moduli_setup.rs");

pub fn main() {
    let x = Bls12381::from_repr(core::array::from_fn(|i| i as u8));
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

    let mut non_reduced = Mersenne61::from_le_bytes_unchecked(&[0xff; 32]);
    assert!(!non_reduced.is_reduced());
    let reduced = &non_reduced + &Mersenne61::ZERO;
    reduced.assert_reduced();

    assert_eq!(&non_reduced + &non_reduced, reduced.double());

    non_reduced = Mersenne61::from_le_bytes_unchecked(&Mersenne61::MODULUS);
    assert!(!non_reduced.is_reduced());

    let mut bytes = [0u8; 32];
    bytes[7] = 1 << 5; // 2^61 = 2^{8*7 + 5} = modulus + 1
    non_reduced = Mersenne61::from_le_bytes_unchecked(&bytes);
    assert!(!non_reduced.is_reduced());
}
