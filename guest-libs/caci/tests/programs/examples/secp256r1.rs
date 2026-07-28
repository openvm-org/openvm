#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use hex_literal::hex;
use openvm_caci::types::{
    ZkvmSecp256r1Hash, ZkvmSecp256r1Pubkey, ZkvmSecp256r1Signature, ZkvmStatus,
};
// clippy thinks this is unused, but it's used in the init! macro
#[allow(unused)]
use openvm_p256::P256Point;

openvm::init!("openvm_init_secp256r1.rs");

openvm::entry!(main);

extern "C" {
    fn zkvm_secp256r1_verify(
        msg: *const ZkvmSecp256r1Hash,
        sig: *const ZkvmSecp256r1Signature,
        pubkey: *const ZkvmSecp256r1Pubkey,
        verified: *mut bool,
    ) -> ZkvmStatus;
}

// Deterministic P-256 test vector: a signature over sha256("caci p256 test message"), generated
// and verified against reference ECDSA math.
const PREHASH: [u8; 32] = hex!("eae25ae81c3d910f690f08ff1b2875b3d544217bbecae2129ec592db535fadaf");
const SIG: [u8; 64] = hex!(
    "2289e972a568c7432584fbddd34d30dcf6b7e97f6e41d211f086516567b2589c
     21f16aef8dd762066876ff6af8aef34445b04d3d549b6540ff38ed62e4c5b024"
);
const PK: [u8; 64] = hex!(
    "48f2ed80370b99e9b8ac2ea8a03b4810145567b6d9e14838ccf312442b9c2042
     f6f0ccae6425b91849e36460c298bdb570b85db57d2268fe905107cf8b9c9d33"
);
/// `n - s` for SIG's `s`: the high-s counterpart, which P-256 accepts.
const SIG_HIGH_S: [u8; 32] =
    hex!("de0e950f72289dfa9789009507510cbb7736ad70527c3943f480dd60179d752d");
/// The secp256r1 group order `n` (not a valid signature scalar).
const ORDER: [u8; 32] = hex!("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551");
/// The secp256r1 field prime `p` (not a canonical coordinate encoding).
const FIELD_P: [u8; 32] = hex!("ffffffff00000001000000000000000000000000ffffffffffffffffffffffff");

fn verify_c(msg: &[u8; 32], sig: &[u8; 64], pk: &[u8; 64]) -> (ZkvmStatus, bool) {
    let msg = ZkvmSecp256r1Hash { data: *msg };
    let sig = ZkvmSecp256r1Signature { data: *sig };
    let pk = ZkvmSecp256r1Pubkey { data: *pk };
    let mut verified = false;
    let status = unsafe { zkvm_secp256r1_verify(&msg, &sig, &pk, &mut verified) };
    (status, verified)
}

pub fn main() {
    // A valid signature verifies.
    let (status, verified) = verify_c(&PREHASH, &SIG, &PK);
    assert!(matches!(status, ZkvmStatus::Ok) && verified);

    // The high-s counterpart also verifies.
    let mut high_s = SIG;
    high_s[32..].copy_from_slice(&SIG_HIGH_S);
    let (status, verified) = verify_c(&PREHASH, &high_s, &PK);
    assert!(matches!(status, ZkvmStatus::Ok) && verified);

    // A corrupted (but still canonical) signature fails verification, not decoding.
    let mut bad_sig = SIG;
    bad_sig[31] ^= 0x01;
    let (status, verified) = verify_c(&PREHASH, &bad_sig, &PK);
    assert!(matches!(status, ZkvmStatus::Ok) && !verified);

    // A different message fails verification.
    let mut other_msg = PREHASH;
    other_msg[0] ^= 0x01;
    let (status, verified) = verify_c(&other_msg, &SIG, &PK);
    assert!(matches!(status, ZkvmStatus::Ok) && !verified);

    // s = 0 and s = n are not valid signature scalars.
    let mut zero_s = SIG;
    zero_s[32..].fill(0);
    assert!(matches!(verify_c(&PREHASH, &zero_s, &PK).0, ZkvmStatus::Fail));
    let mut order_s = SIG;
    order_s[32..].copy_from_slice(&ORDER);
    assert!(matches!(
        verify_c(&PREHASH, &order_s, &PK).0,
        ZkvmStatus::Fail
    ));

    // An off-curve public key (y + 1) fails to decode.
    let mut off_curve = PK;
    off_curve[63] = off_curve[63].wrapping_add(1);
    assert!(matches!(
        verify_c(&PREHASH, &SIG, &off_curve).0,
        ZkvmStatus::Fail
    ));

    // A non-canonical coordinate (x = p) fails to decode.
    let mut bad_coord = PK;
    bad_coord[..32].copy_from_slice(&FIELD_P);
    assert!(matches!(
        verify_c(&PREHASH, &SIG, &bad_coord).0,
        ZkvmStatus::Fail
    ));

    // NULL pointers fail.
    let msg = ZkvmSecp256r1Hash { data: PREHASH };
    let sig = ZkvmSecp256r1Signature { data: SIG };
    let pk = ZkvmSecp256r1Pubkey { data: PK };
    let mut verified = false;
    unsafe {
        use core::ptr;
        assert!(matches!(
            zkvm_secp256r1_verify(ptr::null(), &sig, &pk, &mut verified),
            ZkvmStatus::Fail
        ));
        assert!(matches!(
            zkvm_secp256r1_verify(&msg, ptr::null(), &pk, &mut verified),
            ZkvmStatus::Fail
        ));
        assert!(matches!(
            zkvm_secp256r1_verify(&msg, &sig, ptr::null(), &mut verified),
            ZkvmStatus::Fail
        ));
        assert!(matches!(
            zkvm_secp256r1_verify(&msg, &sig, &pk, ptr::null_mut()),
            ZkvmStatus::Fail
        ));
    }
}
