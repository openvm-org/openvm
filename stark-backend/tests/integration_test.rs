#![feature(trait_upcasting)]
#![allow(incomplete_features)]

use afs_stark_backend::chip::Chip;
/// Test utils
use ax_sdk::{
    any_rap_arc_vec, config,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, FriParameters},
    dummy_airs::{
        fib_air::chip::FibonacciChip,
        interaction::dummy_interaction_air::{DummyInteractionChip, DummyInteractionData},
    },
    engine::StarkFriEngine,
    utils,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_uni_stark::StarkGenericConfig;

mod cached_lookup;
mod fib_selector_air;
mod fib_triples_air;
pub mod interaction;
mod partitioned_sum_air;

#[test]
fn test_single_fib_stark() {
    use ax_sdk::dummy_airs::fib_air::{air::FibonacciAir, trace::generate_trace_rows};

    let log_trace_degree = 3;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let pis = [a, b, get_fib_number(n)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();
    let air = FibonacciAir;

    let trace = generate_trace_rows::<Val>(a, b, n);

    BabyBearPoseidon2Engine::run_simple_test_fast(any_rap_arc_vec![air], vec![trace], vec![pis])
        .expect("Verification failed");
}

#[test]
fn test_single_fib_triples_stark() {
    use fib_triples_air::{air::FibonacciAir, trace::generate_trace_rows};

    let log_trace_degree = 3;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let pis = [a, b, get_fib_number(n + 1)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();

    let air = FibonacciAir;

    let trace = generate_trace_rows::<Val>(a, b, n);

    BabyBearPoseidon2Engine::run_simple_test_fast(any_rap_arc_vec![air], vec![trace], vec![pis])
        .expect("Verification failed");
}

#[test]
fn test_single_fib_selector_stark() {
    use fib_selector_air::{air::FibonacciSelectorAir, trace::generate_trace_rows};

    let log_trace_degree = 3;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let sels: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let pis = [a, b, get_conditional_fib_number(&sels)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();

    let air = FibonacciSelectorAir::new(sels, false);

    let trace = generate_trace_rows::<Val>(a, b, air.sels());

    BabyBearPoseidon2Engine::run_simple_test_fast(any_rap_arc_vec![air], vec![trace], vec![pis])
        .expect("Verification failed");
}

#[test]
fn test_double_fib_starks() {
    use ax_sdk::dummy_airs::{fib_air, fib_air::air::FibonacciAir};
    use fib_selector_air::air::FibonacciSelectorAir;

    let log_n1 = 3;
    let log_n2 = 5;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n1 = 1usize << log_n1;
    let n2 = 1usize << log_n2;

    type Val = BabyBear;
    let sels: Vec<bool> = (0..n2).map(|i| i % 2 == 0).collect(); // Evens
    let pis1 = [a, b, get_fib_number(n1)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();
    let pis2 = [a, b, get_conditional_fib_number(&sels)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();

    let air1 = FibonacciAir;
    let air2 = FibonacciSelectorAir::new(sels, false);

    let trace1 = fib_air::trace::generate_trace_rows::<Val>(a, b, n1);
    let trace2 = fib_selector_air::trace::generate_trace_rows::<Val>(a, b, air2.sels());

    BabyBearPoseidon2Engine::run_simple_test_fast(
        any_rap_arc_vec![air1, air2],
        vec![trace1, trace2],
        vec![pis1, pis2],
    )
    .expect("Verification failed");
}

#[test]
fn test_optional_air() {
    use afs_stark_backend::{engine::StarkEngine, prover::types::ProofInput};
    setup_tracing();

    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let fib_chip = FibonacciChip::new(0, 1, 8);
    let mut send_chip1 = DummyInteractionChip::new_without_partition(1, true, 0);
    let mut send_chip2 =
        DummyInteractionChip::new_with_partition(engine.config().pcs(), 1, true, 0);
    let mut recv_chip1 = DummyInteractionChip::new_without_partition(1, false, 0);
    let mut keygen_builder = engine.keygen_builder();
    let fib_chip_id = keygen_builder.add_air(fib_chip.air());
    let send_chip1_id = keygen_builder.add_air(send_chip1.air());
    let send_chip2_id = keygen_builder.add_air(send_chip2.air());
    let recv_chip1_id = keygen_builder.add_air(recv_chip1.air());
    let pk = keygen_builder.generate_pk();
    let prover = engine.prover();
    let verifier = engine.verifier();

    // Case 1: All AIRs are present.
    {
        let mut challenger = engine.new_challenger();
        send_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        send_chip2.load_data(DummyInteractionData {
            count: vec![1, 2, 8],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        recv_chip1.load_data(DummyInteractionData {
            count: vec![2, 4, 12],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        let proof = prover.prove(
            &mut challenger,
            &pk,
            ProofInput {
                per_air: vec![
                    fib_chip.generate_air_proof_input_with_id(fib_chip_id),
                    send_chip1.generate_air_proof_input_with_id(send_chip1_id),
                    send_chip2.generate_air_proof_input_with_id(send_chip2_id),
                    recv_chip1.generate_air_proof_input_with_id(recv_chip1_id),
                ],
            },
        );
        let mut challenger = engine.new_challenger();
        verifier
            .verify(&mut challenger, &pk.get_vk(), &proof)
            .expect("Verification failed");
    }
    // Case 2: The second AIR is not presented.
    {
        let mut challenger = engine.new_challenger();
        send_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        recv_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        let proof = prover.prove(
            &mut challenger,
            &pk,
            ProofInput {
                per_air: vec![
                    send_chip1.generate_air_proof_input_with_id(send_chip1_id),
                    recv_chip1.generate_air_proof_input_with_id(recv_chip1_id),
                ],
            },
        );
        let mut challenger = engine.new_challenger();
        verifier
            .verify(&mut challenger, &pk.get_vk(), &proof)
            .expect("Verification failed");
    }
    // Case 3: Negative - unbalanced interactions.
    {
        let mut challenger = engine.new_challenger();
        recv_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        let proof = prover.prove(
            &mut challenger,
            &pk,
            ProofInput {
                per_air: vec![recv_chip1.generate_air_proof_input_with_id(recv_chip1_id)],
            },
        );
        let mut challenger = engine.new_challenger();
        assert!(verifier
            .verify(&mut challenger, &pk.get_vk(), &proof)
            .is_err());
    }
}

fn get_fib_number(n: usize) -> u32 {
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n - 1 {
        let c = a + b;
        a = b;
        b = c;
    }
    b
}

fn get_conditional_fib_number(sels: &[bool]) -> u32 {
    let mut a = 0;
    let mut b = 1;
    for &s in sels[0..sels.len() - 1].iter() {
        if s {
            let c = a + b;
            a = b;
            b = c;
        }
    }
    b
}
