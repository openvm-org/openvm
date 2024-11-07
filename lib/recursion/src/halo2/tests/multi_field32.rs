use ax_stark_sdk::config::baby_bear_poseidon2_outer::outer_perm;
use axvm_native_compiler::ir::{Builder, SymbolicExt, Witness};
use p3_baby_bear::BabyBear;
use p3_bn254_fr::Bn254Fr;
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_field::{extension::BinomialExtensionField, AbstractField};
use p3_symmetric::Hash;

use crate::{
    challenger::multi_field32::MultiField32ChallengerVariable,
    config::outer::{OuterChallenger, OuterConfig},
    halo2::Halo2Prover,
    OUTER_DIGEST_SIZE,
};

#[test]
fn test_challenger() {
    let perm = outer_perm();
    let mut challenger = OuterChallenger::new(perm).unwrap();
    let a = BabyBear::from_canonical_usize(1);
    let b = BabyBear::from_canonical_usize(2);
    let c = BabyBear::from_canonical_usize(3);
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

    Halo2Prover::mock::<OuterConfig>(10, builder.operations, Witness::default());
}

#[test]
fn test_challenger_sample_ext() {
    let perm = outer_perm();
    let mut challenger = OuterChallenger::new(perm).unwrap();
    let a = BabyBear::from_canonical_usize(1);
    let b = BabyBear::from_canonical_usize(2);
    let c = BabyBear::from_canonical_usize(3);
    let hash = Hash::from([Bn254Fr::TWO; OUTER_DIGEST_SIZE]);
    challenger.observe(hash);
    challenger.observe(a);
    challenger.observe(b);
    challenger.observe(c);
    let gt1: BinomialExtensionField<BabyBear, 4> = challenger.sample_ext_element();
    challenger.observe(a);
    challenger.observe(b);
    challenger.observe(c);
    let gt2: BinomialExtensionField<BabyBear, 4> = challenger.sample_ext_element();

    let mut builder = Builder::<OuterConfig>::default();
    builder.flags.static_only = true;
    let mut challenger = MultiField32ChallengerVariable::new(&mut builder);
    let a = builder.eval(a);
    let b = builder.eval(b);
    let c = builder.eval(c);
    let hash = builder.eval(Bn254Fr::TWO);
    challenger.observe_commitment(&mut builder, [hash]);
    challenger.observe(&mut builder, a);
    challenger.observe(&mut builder, b);
    challenger.observe(&mut builder, c);
    let result1 = challenger.sample_ext(&mut builder);
    challenger.observe(&mut builder, a);
    challenger.observe(&mut builder, b);
    challenger.observe(&mut builder, c);
    let result2 = challenger.sample_ext(&mut builder);

    builder.assert_ext_eq(SymbolicExt::from_f(gt1), result1);
    builder.assert_ext_eq(SymbolicExt::from_f(gt2), result2);

    Halo2Prover::mock::<OuterConfig>(10, builder.operations, Witness::default());
}
