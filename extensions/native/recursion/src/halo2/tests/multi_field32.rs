use openvm_native_compiler::ir::{Builder, ExtConst, Witness};
use openvm_stark_backend::{
    p3_challenger::{CanObserve, CanSample, FieldChallenger},
    p3_field::{
        extension::BinomialExtensionField, max_absorb_injective_limbs, PrimeCharacteristicRing,
    },
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2_root::root_perm, p3_baby_bear::BabyBear, p3_bn254::Bn254,
};
use p3_symmetric::Hash;

use crate::{
    challenger::multi_field32::MultiField32ChallengerVariable,
    config::outer::{OuterChallenger, OuterConfig},
    halo2::{DslOperations, Halo2Prover},
    OUTER_DIGEST_SIZE,
};

#[test]
fn test_challenger() {
    let perm = root_perm();
    let mut challenger = OuterChallenger::new(perm).unwrap();
    let a = BabyBear::from_usize(1);
    let b = BabyBear::from_usize(2);
    let c = BabyBear::from_usize(3);
    challenger.observe(a);
    challenger.observe(b);
    challenger.observe(c);
    let gt1: BabyBear = challenger.sample();
    challenger.observe(a);
    challenger.observe(b);
    challenger.observe(c);
    let gt2: BabyBear = challenger.sample();
    let gt3: BabyBear = challenger.sample();

    let mut builder = Builder::<OuterConfig>::default();
    builder.flags.static_only = true;
    let mut challenger = MultiField32ChallengerVariable::new(&mut builder);
    let a = builder.eval(a);
    let b = builder.eval(b);
    let c = builder.eval(c);
    challenger.observe(&mut builder, a);
    challenger.observe(&mut builder, b);
    challenger.observe(&mut builder, c);
    let result1 = challenger.sample(&mut builder);
    builder.assert_felt_eq(gt1, result1);
    challenger.observe(&mut builder, a);
    challenger.observe(&mut builder, b);
    challenger.observe(&mut builder, c);
    let result2 = challenger.sample(&mut builder);
    builder.assert_felt_eq(gt2, result2);
    let result3 = challenger.sample(&mut builder);
    builder.assert_felt_eq(gt3, result3);

    Halo2Prover::mock::<OuterConfig>(
        10,
        DslOperations {
            operations: builder.operations,
            num_public_values: 0,
        },
        Witness::default(),
    );
}

/// Sample many times in a row with no intermediate observes so that the squeeze buffer is
/// exhausted and `sample` must trigger fresh duplexings. Exercises the
/// `f_squeeze_buffer.is_empty() && inner_output_buffer.is_empty()` branch.
#[test]
fn test_challenger_sample_chain_matches_native() {
    let perm = root_perm();
    let mut native = OuterChallenger::new(perm.clone()).unwrap();
    for i in 0..3u8 {
        native.observe(BabyBear::from_u8(i));
    }
    let n_samples = 16;
    let gt: Vec<BabyBear> = (0..n_samples).map(|_| native.sample()).collect();

    let mut builder = Builder::<OuterConfig>::default();
    builder.flags.static_only = true;
    let mut challenger = MultiField32ChallengerVariable::new(&mut builder);
    for i in 0..3u8 {
        let v = builder.eval(BabyBear::from_u8(i));
        challenger.observe(&mut builder, v);
    }
    for expected in gt {
        let got = challenger.sample(&mut builder);
        builder.assert_felt_eq(expected, got);
    }

    Halo2Prover::mock::<OuterConfig>(
        10,
        DslOperations {
            operations: builder.operations,
            num_public_values: 0,
        },
        Witness::default(),
    );
}

/// Observe enough `F` values to fully fill `absorb_num_f_elms * RATE` so `observe` triggers an
/// in-line flush, then sample. Verifies the auto-flush path is consistent with native.
#[test]
fn test_challenger_full_batch_observe_matches_native() {
    let perm = root_perm();
    let mut native = OuterChallenger::new(perm.clone()).unwrap();
    // OuterChallenger uses RATE=2 (sponge slots per permute). Pull absorb_num_f_elms from p3 so
    // we exactly cross several flush boundaries regardless of the field choice.
    let absorb_num_f_elms = max_absorb_injective_limbs::<BabyBear, Bn254>();
    let rate = 2usize;
    let n = absorb_num_f_elms * rate * 3 + 1;
    for i in 0..n {
        native.observe(BabyBear::from_usize(i));
    }
    let gt: BabyBear = native.sample();

    let mut builder = Builder::<OuterConfig>::default();
    builder.flags.static_only = true;
    let mut challenger = MultiField32ChallengerVariable::new(&mut builder);
    for i in 0..n {
        let v = builder.eval(BabyBear::from_usize(i));
        challenger.observe(&mut builder, v);
    }
    let result = challenger.sample(&mut builder);
    builder.assert_felt_eq(gt, result);

    Halo2Prover::mock::<OuterConfig>(
        10,
        DslOperations {
            operations: builder.operations,
            num_public_values: 0,
        },
        Witness::default(),
    );
}

#[test]
fn test_challenger_sample_ext() {
    let perm = root_perm();
    let mut challenger = OuterChallenger::new(perm).unwrap();
    let a = BabyBear::from_usize(1);
    let b = BabyBear::from_usize(2);
    let c = BabyBear::from_usize(3);
    let hash = Hash::from([Bn254::TWO; OUTER_DIGEST_SIZE]);
    challenger.observe(hash);
    challenger.observe(a);
    challenger.observe(b);
    challenger.observe(c);
    let gt1: BinomialExtensionField<BabyBear, 4> = challenger.sample_algebra_element();
    challenger.observe(a);
    challenger.observe(b);
    challenger.observe(c);
    let gt2: BinomialExtensionField<BabyBear, 4> = challenger.sample_algebra_element();

    let mut builder = Builder::<OuterConfig>::default();
    builder.flags.static_only = true;
    let mut challenger = MultiField32ChallengerVariable::new(&mut builder);
    let a = builder.eval(a);
    let b = builder.eval(b);
    let c = builder.eval(c);
    let hash = builder.eval(Bn254::TWO);
    challenger.observe_commitment(&mut builder, [hash]);
    challenger.observe(&mut builder, a);
    challenger.observe(&mut builder, b);
    challenger.observe(&mut builder, c);
    let result1 = challenger.sample_ext(&mut builder);
    challenger.observe(&mut builder, a);
    challenger.observe(&mut builder, b);
    challenger.observe(&mut builder, c);
    let result2 = challenger.sample_ext(&mut builder);

    builder.assert_ext_eq(gt1.cons(), result1);
    builder.assert_ext_eq(gt2.cons(), result2);

    Halo2Prover::mock::<OuterConfig>(
        10,
        DslOperations {
            operations: builder.operations,
            num_public_values: 0,
        },
        Witness::default(),
    );
}
