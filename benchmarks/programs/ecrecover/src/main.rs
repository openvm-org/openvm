#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use axvm::io::read_vec;
use axvm_ecc::sw::setup_moduli;
use revm_precompile::secp256k1::ec_recover_run;
use revm_primitives::alloy_primitives::Bytes;

axvm::entry!(main);

pub fn main() {
    setup_moduli();
    let expected_address = read_vec();
    for _ in 0..5 {
        let input = read_vec();
        let recovered = ec_recover_run(&Bytes::from(input), 3000).unwrap();
        assert_eq!(recovered.bytes.as_ref(), expected_address);
    }
}
