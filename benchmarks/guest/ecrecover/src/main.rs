use alloy_primitives::Bytes;
#[allow(unused_imports)] // needed by init! macro
use k256::Secp256k1Point;
use openvm::io::read_vec;
// export native keccak
#[allow(unused_imports, clippy::single_component_path_imports)]
use openvm_keccak256::keccak256;
use revm_precompile::secp256k1::ec_recover_run;

openvm::init!();

pub fn main() {
    let expected_address = read_vec();
    for _ in 0..5 {
        let input = read_vec();
        let recovered = ec_recover_run(&Bytes::from(input), 3000).unwrap();
        assert_eq!(recovered.bytes.as_ref(), expected_address);
    }
}
