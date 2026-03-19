use halo2_base::{gates::circuit::CircuitBuilderStage, utils::fs::gen_srs};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine,
    openvm_stark_backend::{
        p3_util::log2_ceil_usize,
        test_utils::{test_system_params_small, FibFixture, TestFixture},
        StarkEngine,
    },
    utils::setup_tracing,
};
use openvm_static_verifier::{StaticVerifierCircuit, StaticVerifierInput, StaticVerifierShape};

const MIN_ROWS: usize = 20;

/// Find the smallest k such that the static verifier circuit fits in a single advice column.
/// Follows the pattern from the Halo2 wrapper `select_k`.
fn select_k(input: &StaticVerifierInput<'_>) -> usize {
    let mut k = 12;
    let mut first_run = true;
    loop {
        let shape = StaticVerifierShape {
            k,
            lookup_bits: k - 1,
            minimum_rows: MIN_ROWS,
            instance_columns: 1,
        };
        let mut builder = StaticVerifierCircuit::builder(CircuitBuilderStage::Keygen, &shape);
        StaticVerifierCircuit::populate(&mut builder, input);
        let params = builder.calculate_params(Some(MIN_ROWS));
        if params.num_advice_per_phase[0] == 1 {
            builder.clear();
            break;
        }
        if first_run {
            k = log2_ceil_usize(builder.statistics().gate.total_advice_per_phase[0] + MIN_ROWS);
        } else {
            k += 1;
        }
        first_run = false;
        builder.clear();
    }
    tracing::info!("Auto-tuned halo2 k={k}");
    k
}

#[test]
fn real_prover_keygen_prove_verify_roundtrip() {
    setup_tracing();
    let engine: BabyBearBn254Poseidon2CpuEngine =
        BabyBearBn254Poseidon2CpuEngine::new(test_system_params_small(2, 8, 3));

    // Use one FibFixture for keygen, a different one for proving.
    // Both have the same AIR shape (same trace height) but different witnesses.
    let (vk, proof_keygen) = FibFixture::new(0, 1, 1 << 5).keygen_and_prove(&engine);
    let (_vk_prove, proof_prove) = FibFixture::new(1, 1, 1 << 5).keygen_and_prove(&engine);

    let keygen_input = StaticVerifierInput {
        config: engine.config(),
        mvk: &vk,
        proof: &proof_keygen,
    };

    // Auto-tune k to the smallest value that fits in 1 advice column
    let k = select_k(&keygen_input);
    let shape = StaticVerifierShape {
        k,
        lookup_bits: k - 1,
        minimum_rows: MIN_ROWS,
        instance_columns: 1,
    };
    let params = gen_srs(k as u32);

    // Keygen with first proof
    let pinning = StaticVerifierCircuit::keygen(&params, &shape, &keygen_input);

    // Prove with second (different) proof
    let prove_input = StaticVerifierInput {
        config: engine.config(),
        mvk: &vk,
        proof: &proof_prove,
    };
    let halo2_proof = StaticVerifierCircuit::prove(&params, &pinning, &shape, &prove_input);

    // Verify
    assert!(StaticVerifierCircuit::verify(
        &params,
        pinning.pk.get_vk(),
        &halo2_proof
    ));
}
