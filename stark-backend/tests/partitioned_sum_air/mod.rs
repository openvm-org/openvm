use std::sync::Arc;

use ax_stark_backend::{
    prover::{
        types::{AirProofInput, AirProofRawInput, ProofInput},
        USE_DEBUG_BUILDER,
    },
    verifier::VerificationError,
};
use ax_stark_sdk::{config::baby_bear_poseidon2::default_engine, engine::StarkEngine};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;
use rand::{rngs::StdRng, SeedableRng};

use crate::utils::generate_random_matrix;

pub mod air;

use self::air::SumAir;

type Val = BabyBear;

// See air.rs for description of SumAir
fn prove_and_verify_sum_air(x: Vec<Val>, ys: Vec<Vec<Val>>) -> Result<(), VerificationError> {
    assert_eq!(x.len(), ys.len());
    let degree = x.len();
    let log_degree = log2_ceil_usize(degree);

    let engine = default_engine(log_degree);

    let x_trace = RowMajorMatrix::new(x, 1);
    let y_width = ys[0].len();
    let y_trace = RowMajorMatrix::new(ys.into_iter().flatten().collect_vec(), y_width);

    let air = Arc::new(SumAir(y_width));

    let mut keygen_builder = engine.keygen_builder();
    let air_id = keygen_builder.add_air(air.clone());
    let pk = keygen_builder.generate_pk();
    let vk = pk.get_vk();

    let prover = engine.prover();
    // Demonstrate y is cached
    let y_data = prover.committer().commit(vec![y_trace.clone()]);
    // Load x normally
    let air_proof_input = AirProofInput {
        air,
        cached_mains_pdata: vec![y_data],
        raw: AirProofRawInput {
            cached_mains: vec![Arc::new(y_trace)],
            common_main: Some(x_trace),
            public_values: vec![],
        },
    };
    let proof_input = ProofInput::new(vec![(air_id, air_proof_input)]);

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, &pk, proof_input);

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = engine.new_challenger();
    let verifier = engine.verifier();
    verifier.verify(&mut challenger, &vk, &proof)
}

#[test]
fn test_partitioned_sum_air_happy_path() {
    let rng = StdRng::seed_from_u64(0);
    let n = 1 << 3;
    let ys = generate_random_matrix::<Val>(rng, n, 5);
    let x: Vec<Val> = ys
        .iter()
        .map(|row| row.iter().fold(Val::zero(), |sum, x| sum + *x))
        .collect();
    prove_and_verify_sum_air(x, ys).expect("Verification failed");
}

#[test]
fn test_partitioned_sum_air_happy_neg() {
    let rng = StdRng::seed_from_u64(0);
    let n = 1 << 3;
    let ys = generate_random_matrix(rng, n, 5);
    let mut x: Vec<Val> = ys
        .iter()
        .map(|row| row.iter().fold(Val::zero(), |sum, x| sum + *x))
        .collect();
    x[0] = Val::zero();
    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        prove_and_verify_sum_air(x, ys),
        Err(VerificationError::OodEvaluationMismatch)
    );
}
