use core::cmp::max;
use std::sync::Arc;

use openvm_stark_backend::engine::StarkEngine;
use openvm_stark_sdk::{
    config::{FriParameters, baby_bear_poseidon2::BabyBearPoseidon2Engine},
    dummy_airs::fib_air::air::FibonacciAir,
    engine::StarkFriEngine,
};
use p3_field::FieldAlgebra;
use stark_backend_v2::{
    DIGEST_SIZE, EF, F,
    keygen::types::{MultiStarkProvingKeyV2, SystemParams},
    proof::{BatchConstraintProof, GkrLayerClaims, GkrProof, Proof, StackingProof, WhirProof},
};

use crate::system::{Preflight, VerifierCircuit};

fn get_fib_number(n: usize) -> u32 {
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n - 1 {
        let c = (a + b) % (15 * (1 << 27) + 1);
        a = b;
        b = c;
    }
    b
}

#[test]
fn test_circuit() {
    let params = SystemParams {
        l_skip: 3,
        n_stack: 15,
        log_blowup: 1,
        k_whir: 2,
        num_whir_queries: 100,
    };
    let log_trace_degree = 15;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    let pis = [a, b, get_fib_number(n)]
        .map(F::from_canonical_u32)
        .to_vec();
    let air = FibonacciAir;
    // FRI params will be ignored later
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let mut keygen_builder = engine.keygen_builder();
    let air_ids = engine.set_up_keygen_builder(&mut keygen_builder, &[Arc::new(air)]);
    assert_eq!(air_ids[0], 0);
    let fib_pk_v1 = keygen_builder.generate_pk();
    let fib_pk = MultiStarkProvingKeyV2::from_v1(params, fib_pk_v1);
    let fib_vk = fib_pk.get_vk();

    let num_airs = 1;
    let n_max = max(log_trace_degree - params.l_skip, 0);
    let n_logup = 0;
    let whir_rounds = (params.n_stack + (1 << params.k_whir) - 1) / (1 << params.k_whir);

    let num_cols_per_air = fib_vk
        .inner
        .per_air
        .iter()
        .map(|avk| avk.params.width.total_width(0))
        .collect::<Vec<_>>();

    let claims_per_layer: Vec<_> = (0..n_logup)
        .map(|_| GkrLayerClaims {
            p_xi_0: EF::ONE,
            p_xi_1: EF::ONE,
            q_xi_0: EF::ONE,
            q_xi_1: EF::ONE,
        })
        .collect();

    let proof = Proof {
        common_main_commit: [21, 22, 23, 24, 25, 26, 27, 28].map(F::from_canonical_u32),
        cached_commitments_per_air: fib_vk
            .inner
            .per_air
            .iter()
            .map(|avk| {
                (0..avk.num_cached_mains())
                    .map(|_| [F::ZERO; DIGEST_SIZE])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        is_optional_air_present: vec![true; num_airs],
        log_heights: vec![log_trace_degree as u8],
        gkr_proof: GkrProof {
            q0_claim: EF::ONE,
            claims_per_layer,
            sumcheck_polys: (0..n_logup)
                .map(|r| vec![[EF::ZERO; 3]; r])
                .collect::<Vec<_>>(),
        },
        batch_constraint_proof: BatchConstraintProof {
            numerator_term_per_air: vec![EF::ONE; num_airs],
            denominator_term_per_air: vec![EF::ONE; num_airs],
            univariate_round_coeffs: vec![EF::ONE; 1024 * 3],
            sumcheck_round_polys: vec![vec![EF::ONE; 3]; n_max],
            column_openings: num_cols_per_air
                .iter()
                .map(|&num_cols| vec![vec![(EF::ONE, EF::ONE); num_cols]])
                .collect::<Vec<_>>(),
        },
        stacking_proof: StackingProof {
            univariate_round_coeffs: vec![EF::ONE; 1024 * 3],
            sumcheck_round_polys: vec![[EF::ONE; 2]; params.n_stack],
            stacking_openings: vec![vec![EF::ONE; 10]],
        },
        whir_proof: WhirProof {
            whir_sumcheck_polys: vec![[EF::ONE; 2]; params.n_stack + params.l_skip],
            codeword_commits: vec![[F::ZERO; DIGEST_SIZE]; whir_rounds - 1],
            ood_values: vec![EF::ONE; whir_rounds],
            initial_round_opened_rows: vec![vec![
                vec![F::ZERO; 10 * (1 << params.k_whir)];
                params.num_whir_queries
            ]],
            initial_round_merkle_proofs: vec![vec![
                vec![[F::ZERO; DIGEST_SIZE]];
                params.num_whir_queries
            ]],
            codeword_opened_rows: vec![
                vec![
                    vec![EF::ZERO; 1 << params.k_whir];
                    params.num_whir_queries
                ];
                whir_rounds - 1
            ],
            codeword_merkle_proofs: vec![vec![
                vec![[F::ZERO; DIGEST_SIZE]; params.num_whir_queries];
                whir_rounds - 1
            ]],
        },
    };

    let circuit = VerifierCircuit::new();

    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let mut keygen_builder = engine.keygen_builder();

    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();

    let preflight = Preflight::run(&fib_vk, &proof);
    let proof_inputs = circuit.generate_proof_inputs(&fib_vk, &proof, &[pis], &preflight);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}
