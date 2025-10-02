use core::cmp::max;
use std::sync::Arc;

use openvm_stark_backend::{engine::StarkEngine, prover::types::AirProvingContext};
use openvm_stark_sdk::{
    config::{FriParameters, baby_bear_poseidon2::BabyBearPoseidon2Engine},
    dummy_airs::fib_air::air::FibonacciAir,
    engine::StarkFriEngine,
};
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{
    F,
    keygen::types::{MultiStarkProvingKeyV2, SystemParams},
    poseidon2::sponge::DuplexSponge,
    prover::{AirProvingContextV2, ProvingContextV2, prove},
};

use crate::system::VerifierCircuit;

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

pub fn generate_trace_rows<F: PrimeField32>(a: u32, b: u32, n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut rows = vec![vec![F::from_canonical_u32(a), F::from_canonical_u32(b)]];

    for i in 1..n {
        rows.push(vec![rows[i - 1][1], rows[i - 1][0] + rows[i - 1][1]]);
    }

    RowMajorMatrix::new(rows.concat(), 2)
}

#[test]
fn test_circuit() {
    let params = SystemParams {
        l_skip: 2,
        n_stack: 5,
        log_blowup: 1,
        k_whir: 2,
        num_whir_queries: 5, // TEST
    };
    let log_trace_degree = 3;

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

    let trace = generate_trace_rows::<F>(a, b, n);

    let single_ctx = AirProvingContextV2::from_v1(
        params,
        AirProvingContext::simple(Arc::new(trace), pis.clone()),
    );
    let ctx = ProvingContextV2::new(vec![(air_ids[0], single_ctx)]);

    let mut transcript = DuplexSponge::default();
    let proof = prove(&mut transcript, &fib_pk, ctx);

    let circuit = VerifierCircuit::new();

    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let mut keygen_builder = engine.keygen_builder();

    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();

    let proof_inputs = circuit.generate_proof_inputs(&fib_vk, &proof);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}
