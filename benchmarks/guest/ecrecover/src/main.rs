use alloy_primitives::{Bytes, B256};
use openvm::io::read_vec;
// export native keccak
#[allow(unused_imports, clippy::single_component_path_imports)]
use openvm_keccak256::keccak256;
#[allow(unused_imports)] // needed by init! macro
use p256::P256Point;
use revm_precompile::secp256r1::p256_verify;

openvm::init!();

pub fn main() {
    let input = read_vec();
    let target_gas = 3_500u64;
    let outcome = p256_verify(&Bytes::from(input), target_gas).unwrap();
    assert_eq!(outcome.gas_used, 3_450u64);
    let expected_result: Bytes = B256::with_last_byte(1).into();
    assert_eq!(outcome.bytes, expected_result);
}
