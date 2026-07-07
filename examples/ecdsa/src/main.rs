use hex_literal::hex;
use openvm as _;
use openvm_p256::ecdsa::{signature::hazmat::PrehashVerifier, Signature, VerifyingKey};

openvm::init!();

fn main() {
    // Adapted from the FIPS 186-4 ECDSA test vectors for P-256 with SHA-384.
    let public_key = hex!(
        "04
         e0e7b99bc62d8dd67883e39ed9fa0657789c5ff556cc1fd8dd1e2a55e9e3f243
         63fbfd0232b95578075c903a4dbf85ad58f8350516e1ec89b0ee1f5e1362da69"
    );
    let verifier = VerifyingKey::from_sec1_bytes(&public_key).unwrap();
    let signature = Signature::from_scalars(
        hex!("f5087878e212b703578f5c66f434883f3ef414dc23e2e8d8ab6a8d159ed5ad83"),
        hex!("306b4c6c20213707982dffbb30fba99b96e792163dd59dbe606e734328dd7c8a"),
    )
    .unwrap();
    let prehash = hex!(
        "d9c83b92fa0979f4a5ddbd8dd22ab9377801c3c31bf50f932ace0d2146e2574da0d5552dbed4b18836280e9f94558ea6"
    );
    verifier.verify_prehash(&prehash, &signature).unwrap();
}
