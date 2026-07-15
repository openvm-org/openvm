#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use openvm_algebra_guest::{DivUnsafe, IntMod};

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_declare! {
    BlsBase {
        modulus = "0x1A0111EA397FE69A4B1BA7B6434BACD7
                   64774B84F38512BF6730D2A0F6B0F624
                   1EABFFFEB153FFFFB9FEFFFFFFFFAAAB"
    }
}

openvm_algebra_complex_macros::complex_declare! {
    BlsFp2 { mod_type = BlsBase }
}

openvm::init!("openvm_init_rvr_fp2_48.rs");

pub fn main() {
    let mut a = BlsFp2::new(BlsBase::from_u32(7), BlsBase::from_u32(11));
    let mut b = BlsFp2::new(BlsBase::from_u32(13), BlsBase::from_u32(17));

    for _ in 0..8 {
        let add = &a + &b;
        let sub = &add - &b;
        let mul = &a * &b;
        let div = mul.div_unsafe(&b);
        if sub != a || div != a {
            panic!();
        }
        a = add;
        b += &BlsFp2::new(BlsBase::ONE, BlsBase::ONE);
    }
}
