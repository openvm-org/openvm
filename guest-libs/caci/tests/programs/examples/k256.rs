#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use hex_literal::hex;
use openvm_caci::types::{
    ZkvmSecp256k1Hash, ZkvmSecp256k1Pubkey, ZkvmSecp256k1Signature, ZkvmStatus,
};
// clippy thinks this is unused, but it's used in the init! macro
#[allow(unused)]
use openvm_k256::Secp256k1Point;

openvm::init!("openvm_init_k256.rs");

openvm::entry!(main);

extern "C" {
    fn zkvm_secp256k1_verify(
        msg: *const ZkvmSecp256k1Hash,
        sig: *const ZkvmSecp256k1Signature,
        pubkey: *const ZkvmSecp256k1Pubkey,
        verified: *mut bool,
    ) -> ZkvmStatus;
    fn zkvm_secp256k1_ecrecover(
        msg: *const ZkvmSecp256k1Hash,
        sig: *const ZkvmSecp256k1Signature,
        recid: u8,
        output: *mut ZkvmSecp256k1Pubkey,
    ) -> ZkvmStatus;
}

// Test vectors adapted from guest-libs/k256/tests/programs/examples/ecdsa.rs: signatures over
// sha256("example message"), public keys decompressed to uncompressed x || y form, all values
// re-verified against reference ECDSA math.
const PREHASH: [u8; 32] = hex!("ad84cd0b10fc028738971b078124aec2a0e7c6d986a381be0b386f32bee887af");
const SIG0: [u8; 64] = hex!(
    "ce53abb3721bafc561408ce8ff99c909f7f0b18a2f788649d6470162ab1aa032
     3971edc523a6d6453f3fb6128d318d9db1a5ff3386feb1047d9816e780039d52"
);
const PK0: [u8; 64] = hex!(
    "1a7a569e91dbf60581509c7fc946d1003b60c7dee85299538db6353538d59574
     b4e89d60c7d584d084632d296f125f165b4df8e061a49daeba51d36133d03e1a"
);
const SIG1: [u8; 64] = hex!(
    "46c05b6368a44b8810d79859441d819b8e7cdc8bfd371e35c53196f4bcacdb51
     35c7facce2a97b95eacba8a586d87b7958aaf8368ab29cee481f76e871dbd9cb"
);
const PK1: [u8; 64] = hex!(
    "6d6caac248af96f6afa7f904f550253a0f3ef3f5aa2fe6838a95b216691468e2
     487e6222a6664e079c8edf7518defd562dbeda1e7593dfd7f0be285880a24dab"
);
/// `n - s` for SIG0's `s`: the high-s (malleable) counterpart of SIG0.
const SIG0_HIGH_S: [u8; 32] =
    hex!("c68e123adc5929bac0c049ed72ce72610908ddb32849ef37423a47a55032a3ef");
/// The secp256k1 group order `n` (not a valid signature scalar).
const ORDER: [u8; 32] = hex!("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141");
/// The secp256k1 field prime `p` (not a canonical coordinate encoding).
const FIELD_P: [u8; 32] = hex!("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f");

fn verify_c(msg: &[u8; 32], sig: &[u8; 64], pk: &[u8; 64]) -> (ZkvmStatus, bool) {
    let msg = ZkvmSecp256k1Hash { data: *msg };
    let sig = ZkvmSecp256k1Signature { data: *sig };
    let pk = ZkvmSecp256k1Pubkey { data: *pk };
    let mut verified = false;
    let status = unsafe { zkvm_secp256k1_verify(&msg, &sig, &pk, &mut verified) };
    (status, verified)
}

fn ecrecover_c(msg: &[u8; 32], sig: &[u8; 64], recid: u8) -> (ZkvmStatus, [u8; 64]) {
    let msg = ZkvmSecp256k1Hash { data: *msg };
    let sig = ZkvmSecp256k1Signature { data: *sig };
    let mut output = ZkvmSecp256k1Pubkey { data: [0u8; 64] };
    let status = unsafe { zkvm_secp256k1_ecrecover(&msg, &sig, recid, &mut output) };
    (status, output.data)
}

fn with_high_s(sig: &[u8; 64]) -> [u8; 64] {
    let mut out = *sig;
    out[32..].copy_from_slice(&SIG0_HIGH_S);
    out
}

pub fn main() {
    // Valid signatures verify.
    let (status, verified) = verify_c(&PREHASH, &SIG0, &PK0);
    assert!(matches!(status, ZkvmStatus::Ok) && verified);
    let (status, verified) = verify_c(&PREHASH, &SIG1, &PK1);
    assert!(matches!(status, ZkvmStatus::Ok) && verified);

    // A corrupted (but still canonical) signature fails verification, not decoding.
    let mut bad_sig = SIG0;
    bad_sig[31] ^= 0x01;
    let (status, verified) = verify_c(&PREHASH, &bad_sig, &PK0);
    assert!(matches!(status, ZkvmStatus::Ok) && !verified);

    // A wrong-key verification fails.
    let (status, verified) = verify_c(&PREHASH, &SIG0, &PK1);
    assert!(matches!(status, ZkvmStatus::Ok) && !verified);

    // The high-s counterpart is a valid scalar pair but is rejected as malleable (k256 policy).
    let (status, verified) = verify_c(&PREHASH, &with_high_s(&SIG0), &PK0);
    assert!(matches!(status, ZkvmStatus::Ok) && !verified);

    // s = 0 and s = n are not valid signature scalars.
    let mut zero_s = SIG0;
    zero_s[32..].fill(0);
    assert!(matches!(
        verify_c(&PREHASH, &zero_s, &PK0).0,
        ZkvmStatus::Fail
    ));
    let mut order_s = SIG0;
    order_s[32..].copy_from_slice(&ORDER);
    assert!(matches!(
        verify_c(&PREHASH, &order_s, &PK0).0,
        ZkvmStatus::Fail
    ));

    // An off-curve public key (y + 1) fails to decode.
    let mut off_curve = PK0;
    off_curve[63] = off_curve[63].wrapping_add(1);
    assert!(matches!(
        verify_c(&PREHASH, &SIG0, &off_curve).0,
        ZkvmStatus::Fail
    ));

    // A non-canonical coordinate (x = p) fails to decode.
    let mut bad_coord = PK0;
    bad_coord[..32].copy_from_slice(&FIELD_P);
    assert!(matches!(
        verify_c(&PREHASH, &SIG0, &bad_coord).0,
        ZkvmStatus::Fail
    ));

    // Recovery returns the signing key.
    let (status, recovered) = ecrecover_c(&PREHASH, &SIG0, 0);
    assert!(matches!(status, ZkvmStatus::Ok));
    assert_eq!(recovered, PK0);
    let (status, recovered) = ecrecover_c(&PREHASH, &SIG1, 1);
    assert!(matches!(status, ZkvmStatus::Ok));
    assert_eq!(recovered, PK1);

    // Unlike verification, recovery accepts high-s signatures (EVM precompile behavior); the
    // y-parity flips with the negated s.
    let (status, recovered) = ecrecover_c(&PREHASH, &with_high_s(&SIG0), 1);
    assert!(matches!(status, ZkvmStatus::Ok));
    assert_eq!(recovered, PK0);

    // recid > 3 is invalid; recid 2/3 (x-reduced) fails here because r + n overflows the field.
    assert!(matches!(
        ecrecover_c(&PREHASH, &SIG0, 4).0,
        ZkvmStatus::Fail
    ));
    assert!(matches!(
        ecrecover_c(&PREHASH, &SIG0, 2).0,
        ZkvmStatus::Fail
    ));

    // r = 0 is not a valid signature scalar.
    let mut zero_r = SIG0;
    zero_r[..32].fill(0);
    assert!(matches!(
        ecrecover_c(&PREHASH, &zero_r, 0).0,
        ZkvmStatus::Fail
    ));

    // NULL pointers fail.
    let msg = ZkvmSecp256k1Hash { data: PREHASH };
    let sig = ZkvmSecp256k1Signature { data: SIG0 };
    let pk = ZkvmSecp256k1Pubkey { data: PK0 };
    let mut verified = false;
    let mut output = ZkvmSecp256k1Pubkey { data: [0u8; 64] };
    unsafe {
        use core::ptr;
        assert!(matches!(
            zkvm_secp256k1_verify(ptr::null(), &sig, &pk, &mut verified),
            ZkvmStatus::Fail
        ));
        assert!(matches!(
            zkvm_secp256k1_verify(&msg, ptr::null(), &pk, &mut verified),
            ZkvmStatus::Fail
        ));
        assert!(matches!(
            zkvm_secp256k1_verify(&msg, &sig, ptr::null(), &mut verified),
            ZkvmStatus::Fail
        ));
        assert!(matches!(
            zkvm_secp256k1_verify(&msg, &sig, &pk, ptr::null_mut()),
            ZkvmStatus::Fail
        ));
        assert!(matches!(
            zkvm_secp256k1_ecrecover(ptr::null(), &sig, 0, &mut output),
            ZkvmStatus::Fail
        ));
        assert!(matches!(
            zkvm_secp256k1_ecrecover(&msg, ptr::null(), 0, &mut output),
            ZkvmStatus::Fail
        ));
        assert!(matches!(
            zkvm_secp256k1_ecrecover(&msg, &sig, 0, ptr::null_mut()),
            ZkvmStatus::Fail
        ));
    }
}
