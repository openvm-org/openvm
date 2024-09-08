use std::str::FromStr;

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

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;
fn run_test_program(test_program: impl FnOnce(&mut AsmBuilder<F, EF>)) {
    let mut builder = AsmBuilder::<F, EF>::bigint_builder();
    test_program(&mut builder);
    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

fn test_verify_single_ecdsa(input: ECDSAInput) {
    run_test_program(move |builder: &mut AsmBuilder<F, EF>| {
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
        let verify_result = verify_ecdsa_secp256k1(builder, &input_var, 4);
        builder.assert_var_eq(verify_result, F::one());
        builder.halt();
    });
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
fn test_ecdsa_verify_happy_path() {
    for seed in [42, 13, 24] {
        test_verify_single_ecdsa(get_test_ecdsa_input(seed));
    }
}

#[test]
#[should_panic]
fn test_ecdsa_verify_negative() {
    for seed in [42, 13, 24] {
        let mut input = get_test_ecdsa_input(seed);
        input.msg_hash += 1u64;
        test_verify_single_ecdsa(input);
    }
}

#[test]
fn test_ec_point_verify() {
    run_test_program(move |builder: &mut AsmBuilder<F, EF>| {
        // Point on curve.
        let point1 = ECPointVariable {
            x: builder.eval_biguint(
                BigUint::from_str(
                    "55066263022277343669578718895168534326250603453777594175500187360389116729240",
                )
                .unwrap(),
            ),
            y: builder.eval_biguint(
                BigUint::from_str(
                    "32670510020758816978083085130507043184471273380659243275938904335757337482424",
                )
                .unwrap(),
            ),
        };
        let point1_valid = point1.is_valid(builder);
        builder.assert_var_eq(point1_valid, F::one());

        // Point (2,1) not on curve.
        let point2 = ECPointVariable {
            x: builder.eval_biguint(BigUint::from_str("2").unwrap()),
            y: builder.eval_biguint(BigUint::from_str("1").unwrap()),
        };
        let point2_valid = point2.is_valid(builder);
        builder.assert_var_eq(point2_valid, F::zero());

        // Identity point is valid.
        let point3 = ECPointVariable {
            x: builder.eval_biguint(BigUint::from_str("0").unwrap()),
            y: builder.eval_biguint(BigUint::from_str("0").unwrap()),
        };
        let point3_valid = point3.is_valid(builder);
        builder.assert_var_eq(point3_valid, F::one());
        builder.halt();
    });
}

#[test]
fn test_ecdsa_signature_verify() {
    run_test_program(move |builder: &mut AsmBuilder<F, EF>| {
        // Invalid because r == 0.
        let sig1 = ECDSASignatureVariable {
            r: builder.eval_biguint(BigUint::from(0u64)),
            s: builder.eval_biguint(BigUint::from(1u64)),
        };
        let sig1_is_valid = sig1.is_valid(builder);
        builder.assert_var_eq(sig1_is_valid, F::zero());
        // Invalid because s == 0.
        let sig2 = ECDSASignatureVariable {
            r: builder.eval_biguint(BigUint::from(1u64)),
            s: builder.eval_biguint(BigUint::from(0u64)),
        };
        let sig2_is_valid = sig2.is_valid(builder);
        builder.assert_var_eq(sig2_is_valid, F::zero());
        // Valid.
        let sig3 = ECDSASignatureVariable {
            r: builder.eval_biguint(BigUint::from(1u64)),
            s: builder.eval_biguint(BigUint::from(1u64)),
        };
        let sig3_is_valid = sig3.is_valid(builder);
        builder.assert_var_eq(sig3_is_valid, F::one());
        builder.halt();
    });
}
