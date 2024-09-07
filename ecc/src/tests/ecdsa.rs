use afs_compiler::{asm::AsmBuilder, util::execute_program};
use k256::{
    ecdsa::{hazmat::DigestPrimitive, signature::Signer, Signature, SigningKey, VerifyingKey},
    sha2::digest::FixedOutput,
    Secp256k1,
};
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use rand::{rngs::StdRng, SeedableRng};
use sha3::Digest;

use crate::{
    ecdsa::verify_ecdsa_secp256k1,
    types::{
        ECDSAInput, ECDSAInputVariable, ECDSASignature, ECDSASignatureVariable, ECPoint,
        ECPointVariable,
    },
};

fn test_verify_single_ecdsa(input: ECDSAInput) {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::bigint_builder();

    let ECDSAInput {
        pubkey,
        sig,
        msg_hash,
    } = input;

    let input_var = ECDSAInputVariable {
        pubkey: ECPointVariable {
            x: builder.eval_biguint(pubkey.x),
            y: builder.eval_biguint(pubkey.y),
        },
        sig: ECDSASignatureVariable {
            r: builder.eval_biguint(sig.r),
            s: builder.eval_biguint(sig.s),
        },
        msg_hash: builder.eval_biguint(msg_hash),
    };
    let verify_result = verify_ecdsa_secp256k1(&mut builder, &input_var, 4);
    builder.assert_var_eq(verify_result, F::one());
    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

fn get_test_ecdsa_input(seed: u64) -> ECDSAInput {
    let mut rng = StdRng::seed_from_u64(seed);
    // Signing
    let signing_key = SigningKey::random(&mut rng);

    let message = b"ECDSA proves knowledge of a secret number in the context of a single message";
    let sig: Signature = signing_key.sign(message);
    let sig: ECDSASignature = sig.into();

    let msg_hash = <Secp256k1 as DigestPrimitive>::Digest::new_with_prefix(message);
    let msg_hash = BigUint::from_bytes_be(msg_hash.finalize_fixed().as_slice());

    // Verification
    let verifying_key = VerifyingKey::from(&signing_key);
    let pubkey: ECPoint = verifying_key.into();

    ECDSAInput {
        pubkey,
        sig,
        msg_hash,
    }
}

#[test]
fn test_ecdsa_happy_path() {
    for seed in [42, 13, 24] {
        test_verify_single_ecdsa(get_test_ecdsa_input(seed));
    }
}

#[test]
#[should_panic]
fn test_ecdsa_negative() {
    for seed in [42, 13, 24] {
        let mut input = get_test_ecdsa_input(seed);
        input.msg_hash += 1u64;
        test_verify_single_ecdsa(input);
    }
}
