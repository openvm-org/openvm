use std::sync::Arc;

use halo2_base::{
    gates::{
        circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage},
        GateInstructions, RangeInstructions,
    },
    halo2_proofs::dev::MockProver,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        default_transcript, BabyBearBn254Poseidon2Config as RootConfig, Bn254Scalar,
        D_EF as NATIVE_EF_DEGREE,
    },
    openvm_stark_backend::{
        p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64},
        FiatShamirTranscript,
    },
};

use super::*;
use crate::{
    config::{STATIC_VERIFIER_LOOKUP_ADVICE_COLS, STATIC_VERIFIER_NUM_ADVICE_COLS},
    field::baby_bear::{BabyBearChip, BabyBearExt4Wire},
    RootEF, RootF,
};

fn run_mock(expect_satisfied: bool, build: impl FnOnce(&mut BaseCircuitBuilder<Fr>)) {
    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(17)
        .use_lookup_bits(8)
        .use_instance_columns(1);
    build(&mut builder);

    let params = builder.calculate_params(Some(4096));
    assert!(
        params
            .num_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default()
            >= STATIC_VERIFIER_NUM_ADVICE_COLS
    );
    assert!(
        params
            .num_lookup_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default()
            >= STATIC_VERIFIER_LOOKUP_ADVICE_COLS
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

fn ext_to_u64(ext: RootEF) -> [u64; NATIVE_EF_DEGREE] {
    core::array::from_fn(|i| {
        <RootEF as BasedVectorSpace<RootF>>::as_basis_coefficients_slice(&ext)[i].as_canonical_u64()
    })
}

#[test]
fn transcript_outputs_match_native_interleaved_flow() {
    let observed_ext_coeffs = [5, 7, 11, 13];
    let digest = [Bn254Scalar::from_u64(0x1234_5678)];

    // Build transcript state up to the PoW check, then grind for a valid witness.
    let build_transcript_before_pow = || {
        let mut t = default_transcript();
        t.observe(RootF::from_u64(1));
        t.observe(RootF::from_u64(2));
        t.observe(RootF::from_u64(3));
        t.observe_ext(RootEF::from_basis_coefficients_fn(|i| {
            RootF::from_u64(observed_ext_coeffs[i])
        }));
        t.observe_commit(digest);
        let _ = t.sample();
        let _ = FiatShamirTranscript::<RootConfig>::sample_ext(&mut t);
        let _ = t.sample_bits(17);
        t
    };
    let witness_for_pow = (0u64..)
        .find(|&w| {
            let mut t = build_transcript_before_pow();
            t.check_witness(9, RootF::from_u64(w))
        })
        .expect("should find a valid PoW witness by grinding");

    let mut native = default_transcript();
    native.observe(RootF::from_u64(1));
    native.observe(RootF::from_u64(2));
    native.observe(RootF::from_u64(3));
    native.observe_ext(RootEF::from_basis_coefficients_fn(|i| {
        RootF::from_u64(observed_ext_coeffs[i])
    }));
    native.observe_commit(digest);

    let expected_sample = native.sample().as_canonical_u64();
    let expected_ext = ext_to_u64(FiatShamirTranscript::<RootConfig>::sample_ext(&mut native));
    let expected_bits = native.sample_bits(17) as u64;
    assert!(native.check_witness(9, RootF::from_u64(witness_for_pow)));
    let expected_followup = native.sample().as_canonical_u64();

    run_mock(true, |builder| {
        let range = builder.range_chip();
        let baby_bear = BabyBearChip::new(Arc::new(range.clone()));

        let ctx = builder.main(0);
        let gate = range.gate();

        let mut transcript = TranscriptGadget::new(ctx);

        let one = baby_bear.load_witness(ctx, RootF::from_u64(1));
        let two = baby_bear.load_witness(ctx, RootF::from_u64(2));
        let three = baby_bear.load_witness(ctx, RootF::from_u64(3));
        transcript.observe(ctx, &baby_bear, &one);
        transcript.observe(ctx, &baby_bear, &two);
        transcript.observe(ctx, &baby_bear, &three);

        let observed_ext = BabyBearExt4Wire(core::array::from_fn(|i| {
            baby_bear.load_witness(ctx, RootF::from_u64(observed_ext_coeffs[i]))
        }));
        transcript.observe_ext(ctx, &baby_bear, &observed_ext);

        let digest_wire = TranscriptGadget::load_digest_witness(ctx, digest);
        transcript.observe_commit(ctx, &baby_bear, &digest_wire);

        let sampled = transcript.sample(ctx, &baby_bear);
        gate.assert_is_const(ctx, &sampled.value, &Fr::from(expected_sample));

        let sampled_ext = transcript.sample_ext(ctx, &baby_bear);
        for (i, coeff) in sampled_ext.0.iter().enumerate() {
            gate.assert_is_const(ctx, &coeff.value, &Fr::from(expected_ext[i]));
        }

        let sampled_bits = transcript.sample_bits(ctx, &baby_bear, 17);
        gate.assert_is_const(ctx, &sampled_bits, &Fr::from(expected_bits));

        let pow_witness = baby_bear.load_witness(ctx, RootF::from_u64(witness_for_pow));
        // check_witness now returns () and asserts internally
        transcript.check_witness(ctx, &baby_bear, 9, &pow_witness);

        let followup = transcript.sample(ctx, &baby_bear);
        gate.assert_is_const(ctx, &followup.value, &Fr::from(expected_followup));
    });
}

#[test]
fn transcript_check_witness_zero_bits_matches_native() {
    let mut native = default_transcript();
    native.observe(RootF::from_u64(99));
    let expected_first = native.sample().as_canonical_u64();
    // check_witness now asserts internally; just call it for its side effect on the transcript
    let _ = native.check_witness(0, RootF::from_u64(7));
    let expected_second = native.sample().as_canonical_u64();

    run_mock(true, |builder| {
        let range = builder.range_chip();
        let baby_bear = BabyBearChip::new(Arc::new(range.clone()));

        let ctx = builder.main(0);
        let gate = range.gate();

        let mut transcript = TranscriptGadget::new(ctx);

        let obs = baby_bear.load_witness(ctx, RootF::from_u64(99));
        transcript.observe(ctx, &baby_bear, &obs);

        let first = transcript.sample(ctx, &baby_bear);
        gate.assert_is_const(ctx, &first.value, &Fr::from(expected_first));

        let witness = baby_bear.load_witness(ctx, RootF::from_u64(7));
        // check_witness now returns () and asserts internally
        transcript.check_witness(ctx, &baby_bear, 0, &witness);

        let second = transcript.sample(ctx, &baby_bear);
        gate.assert_is_const(ctx, &second.value, &Fr::from(expected_second));
    });
}

#[test]
fn transcript_sample_bits_rejects_bits_equal_31() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_mock(true, |builder| {
            let range = builder.range_chip();
            let baby_bear = BabyBearChip::new(Arc::new(range.clone()));
            let ctx = builder.main(0);
            let mut transcript = TranscriptGadget::new(ctx);
            let _ = transcript.sample_bits(ctx, &baby_bear, 31);
        });
    }));
    assert!(
        result.is_err(),
        "sample_bits(31) must be rejected to match backend bound semantics",
    );
}
