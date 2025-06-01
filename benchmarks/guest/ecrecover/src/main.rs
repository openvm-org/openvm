use alloy_primitives::{Bytes, B256, B512};
// Be careful not to import k256::ecdsa::{Signature, VerifyingKey}
// because those are type aliases that their (non-zkvm) implementations
use k256::{
    ecdsa::{Error, RecoveryId, Signature, VerifyingKey},
    Secp256k1Point,
};
use openvm::io::read_vec;
#[allow(unused_imports, clippy::single_component_path_imports)]
use openvm_keccak256::keccak256;
// export native keccak
use revm_precompile::{
    utilities::right_pad, Error as PrecompileError, PrecompileOutput, PrecompileResult,
};

openvm::init!();

pub fn main() {
    let expected_address = read_vec();
    for _ in 0..5 {
        let input = read_vec();
        let recovered = ec_recover_run(&Bytes::from(input), 3000).unwrap();
        assert_eq!(recovered.bytes.as_ref(), expected_address);
    }
}

fn ecrecover(sig: &B512, mut recid: u8, msg: &B256) -> Result<B256, Error> {
    // parse signature
    let mut sig = Signature::from_slice(sig.as_slice())?;
    if let Some(sig_normalized) = sig.normalize_s() {
        sig = sig_normalized;
        recid ^= 1;
    }
    let recid = RecoveryId::from_byte(recid).expect("recovery ID is valid");

    let recovered_key = VerifyingKey::recover_from_prehash(&msg[..], &sig, recid)?;
    let mut hash = keccak256(
        &recovered_key
            .to_encoded_point(/* compress = */ false)
            .as_bytes()[1..],
    );

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
