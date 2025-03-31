use alloy_primitives::{keccak256, Bytes, B256, B512};
use k256::{
    ecdsa::{Error, RecoveryId, Signature},
    Secp256k1,
};
use openvm::io::read_vec;
#[allow(unused_imports)]
use openvm_ecc_guest::{
    algebra::IntMod, ecdsa::VerifyingKey, k256::Secp256k1Point, weierstrass::WeierstrassPoint,
};
#[allow(unused_imports, clippy::single_component_path_imports)]
use openvm_keccak256_guest;
// export native keccak
use revm_precompile::{
    utilities::right_pad, Error as PrecompileError, PrecompileOutput, PrecompileResult,
};

openvm_algebra_guest::moduli_macros::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
}
openvm_ecc_guest::sw_macros::sw_init! {
    Secp256k1Point,
}

pub fn main() {
    setup_all_moduli();
    setup_all_curves();

    let expected_address = read_vec();
    for _ in 0..5 {
        let input = read_vec();
        let recovered = ec_recover_run(&Bytes::from(input), 3000).unwrap();
        assert_eq!(recovered.bytes.as_ref(), expected_address);
    }
}

// OpenVM version of ecrecover precompile.
pub fn ecrecover(sig: &B512, mut recid: u8, msg: &B256) -> Result<B256, Error> {
    // parse signature
    let mut sig = Signature::from_slice(sig.as_slice())?;
    if let Some(sig_normalized) = sig.normalize_s() {
        sig = sig_normalized;
        recid ^= 1;
    }
    let recid = RecoveryId::from_byte(recid).expect("recovery ID is valid");

    // annoying: Signature::to_bytes copies from slice
    let recovered_key =
        VerifyingKey::<Secp256k1>::recover_from_prehash_noverify(&msg[..], &sig.to_bytes(), recid)?;
    let public_key = recovered_key.as_affine();
    let mut encoded = [0u8; 64];
    encoded[..32].copy_from_slice(&public_key.x().to_be_bytes());
    encoded[32..].copy_from_slice(&public_key.y().to_be_bytes());
    // hash it
    let mut hash = keccak256(encoded);
    // truncate to 20 bytes
    hash[..12].fill(0);
    Ok(B256::from(hash))
}

// We replicate code from `revm-precompile` to avoid importing the patched version.
pub fn ec_recover_run(input: &Bytes, gas_limit: u64) -> PrecompileResult {
    const ECRECOVER_BASE: u64 = 3_000;

    if ECRECOVER_BASE > gas_limit {
        return Err(PrecompileError::OutOfGas.into());
    }

    let input = right_pad::<128>(input);

    // `v` must be a 32-byte big-endian integer equal to 27 or 28.
    if !(input[32..63].iter().all(|&b| b == 0) && matches!(input[63], 27 | 28)) {
        return Ok(PrecompileOutput::new(ECRECOVER_BASE, Bytes::new()));
    }

    let msg = <&B256>::try_from(&input[0..32]).unwrap();
    let recid = input[63] - 27;
    let sig = <&B512>::try_from(&input[64..128]).unwrap();

    let out = ecrecover(sig, recid, msg)
        .map(|o| o.to_vec().into())
        .unwrap_or_default();
    Ok(PrecompileOutput::new(ECRECOVER_BASE, out))
}
