#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm::io::read_vec;
use axvm_ecc::pairing::final_exp_hint::bls12_381_final_exp_hint;

axvm::entry!(main);

pub fn main() {
    let io = read_vec();
    let f = &io[..48 * 12];
    let expected = &io[48 * 12..];
    let actual = bls12_381_final_exp_hint(f);
    assert_eq!(&actual, expected);
}
