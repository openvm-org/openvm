use halo2_base::{
    gates::circuit::{CircuitBuilderStage, builder::BaseCircuitBuilder},
    halo2_proofs::dev::MockProver,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        BabyBearBn254Poseidon2Config as NativeConfig, Bn254Scalar, D_EF as NATIVE_EF_DEGREE,
        EF as NativeEF, F as NativeF, default_transcript,
    },
    openvm_stark_backend::{
        FiatShamirTranscript,
        p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64},
    },
};

use crate::config::{
    STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0, STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0,
};

use super::*;

fn run_mock(expect_satisfied: bool, build: impl FnOnce(&mut BaseCircuitBuilder<Fr>)) {
    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(17)
        .use_lookup_bits(8)
        .use_instance_columns(1);
    build(&mut builder);

    let params = builder.calculate_params(Some(4096));
    assert_eq!(
        params
            .num_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default(),
        STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0
    );
    assert_eq!(
        params
            .num_lookup_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default(),
        STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0
    );

    let prover = MockProver::run(17, &builder, vec![vec![]])
        .expect("mock prover should initialize transcript gadget circuit");
    if expect_satisfied {
        prover.assert_satisfied();
    } else {
        assert!(
            prover.verify().is_err(),
            "expected transcript replay constraints to fail"
        );
    }
}

fn ext_to_u64(ext: NativeEF) -> [u64; NATIVE_EF_DEGREE] {
    core::array::from_fn(|i| {
        <NativeEF as BasedVectorSpace<NativeF>>::as_basis_coefficients_slice(&ext)[i]
            .as_canonical_u64()
    })
}

#[test]
fn transcript_outputs_match_native_interleaved_flow() {
    let observed_ext_coeffs = [5, 7, 11, 13];
    let digest = [Bn254Scalar::from_u64(0x1234_5678)];
    let witness_for_pow = 42u64;

    let mut native = default_transcript();
    native.observe(NativeF::from_u64(1));
    native.observe(NativeF::from_u64(2));
    native.observe(NativeF::from_u64(3));
    native.observe_ext(NativeEF::from_basis_coefficients_fn(|i| {
        NativeF::from_u64(observed_ext_coeffs[i])
    }));
    native.observe_commit(digest);

    let expected_sample = native.sample().as_canonical_u64();
    let expected_ext = ext_to_u64(FiatShamirTranscript::<NativeConfig>::sample_ext(
        &mut native,
    ));
    let expected_bits = native.sample_bits(17) as u64;
    let expected_pow = native.check_witness(9, NativeF::from_u64(witness_for_pow));
    let expected_followup = native.sample().as_canonical_u64();

    run_mock(true, |builder| {
        let baby_bear = BabyBearArithmeticGadgets;

        let range = builder.range_chip();
        let ctx = builder.main(0);
        let gate = range.gate();

        let mut transcript = TranscriptGadget::new(ctx);

        let one = baby_bear.load_witness(ctx, &range, 1);
        let two = baby_bear.load_witness(ctx, &range, 2);
        let three = baby_bear.load_witness(ctx, &range, 3);
        transcript.observe(ctx, &range, &baby_bear, &one);
        transcript.observe(ctx, &range, &baby_bear, &two);
        transcript.observe(ctx, &range, &baby_bear, &three);

        let observed_ext = baby_bear.load_ext_witness(ctx, &range, observed_ext_coeffs);
        transcript.observe_ext(ctx, &range, &baby_bear, &observed_ext);

        let digest_var = TranscriptGadget::load_digest_witness(ctx, digest);
        transcript.observe_commit(ctx, &range, &baby_bear, &digest_var);

        let sampled = transcript.sample(ctx, &range, &baby_bear);
        gate.assert_is_const(ctx, &sampled.cell, &Fr::from(expected_sample));

        let sampled_ext = transcript.sample_ext(ctx, &range, &baby_bear);
        for (i, coeff) in sampled_ext.coeffs.iter().enumerate() {
            gate.assert_is_const(ctx, &coeff.cell, &Fr::from(expected_ext[i]));
        }

        let sampled_bits = transcript.sample_bits(ctx, &range, &baby_bear, 17);
        gate.assert_is_const(ctx, &sampled_bits, &Fr::from(expected_bits));

        let pow_witness = baby_bear.load_witness(ctx, &range, witness_for_pow);
        let pow_ok = transcript.check_witness(ctx, &range, &baby_bear, 9, &pow_witness);
        gate.assert_is_const(ctx, &pow_ok, &Fr::from(expected_pow as u64));

        let followup = transcript.sample(ctx, &range, &baby_bear);
        gate.assert_is_const(ctx, &followup.cell, &Fr::from(expected_followup));
    });
}

#[test]
fn transcript_check_witness_zero_bits_matches_native() {
    let mut native = default_transcript();
    native.observe(NativeF::from_u64(99));
    let expected_first = native.sample().as_canonical_u64();
    let expected_check = native.check_witness(0, NativeF::from_u64(7));
    let expected_second = native.sample().as_canonical_u64();

    run_mock(true, |builder| {
        let baby_bear = BabyBearArithmeticGadgets;

        let range = builder.range_chip();
        let ctx = builder.main(0);
        let gate = range.gate();

        let mut transcript = TranscriptGadget::new(ctx);

        let obs = baby_bear.load_witness(ctx, &range, 99);
        transcript.observe(ctx, &range, &baby_bear, &obs);

        let first = transcript.sample(ctx, &range, &baby_bear);
        gate.assert_is_const(ctx, &first.cell, &Fr::from(expected_first));

        let witness = baby_bear.load_witness(ctx, &range, 7);
        let check = transcript.check_witness(ctx, &range, &baby_bear, 0, &witness);
        gate.assert_is_const(ctx, &check, &Fr::from(expected_check as u64));

        let second = transcript.sample(ctx, &range, &baby_bear);
        gate.assert_is_const(ctx, &second.cell, &Fr::from(expected_second));
    });
}

#[test]
fn transcript_sample_bits_rejects_bits_equal_31() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_mock(true, |builder| {
            let baby_bear = BabyBearArithmeticGadgets;
            let range = builder.range_chip();
            let ctx = builder.main(0);
            let mut transcript = TranscriptGadget::new(ctx);
            let _ = transcript.sample_bits(ctx, &range, &baby_bear, 31);
        });
    }));
    assert!(
        result.is_err(),
        "sample_bits(31) must be rejected to match backend bound semantics",
    );
}

#[test]
fn logged_transcript_replay_detects_tampered_sample() {
    let digest = [Bn254Scalar::from_u64(0x9999)];

    let mut logged = LoggedTranscript::new();
    logged.observe(NativeF::from_u64(3));
    logged.observe(NativeF::from_u64(8));
    logged.observe_commit(digest);
    let _ = logged.sample();
    let _ = logged.sample();
    let _ = logged.sample_bits(9);
    let mut events = logged.into_events();

    run_mock(true, |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        constrain_transcript_events(ctx, &range, &events);
    });

    let tampered_idx = events
        .iter()
        .position(|event| matches!(event, TranscriptEvent::Sample(_)))
        .expect("logged transcript should contain sample events");
    let tampered = match events[tampered_idx] {
        TranscriptEvent::Sample(value) => {
            TranscriptEvent::Sample((value + 1) % BABY_BEAR_MODULUS_U64)
        }
        TranscriptEvent::Observe(_) => unreachable!(),
    };
    events[tampered_idx] = tampered;

    run_mock(false, |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        constrain_transcript_events(ctx, &range, &events);
    });
}
