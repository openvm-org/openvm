//! Integration: one Halo2 KZG roundtrip on a BN254 [`MixtureFixture`] proof using only
//! [`StaticVerifierCircuit::populate_verify_stark_constraints`] (no continuations public values or
//! DAG cached-commit pin).
//!
//! Full [`StaticVerifierCircuit::populate`] end-to-end is exercised in `openvm-sdk` integration
//! tests, not here.

use std::sync::Arc;

use halo2_base::{
    gates::circuit::CircuitBuilderStage,
    halo2_proofs::plonk::{keygen_pk, keygen_vk},
    utils::fs::gen_srs,
};
use openvm_stark_backend::{
    p3_util::log2_ceil_usize,
    proof::Proof,
    test_utils::{test_system_params_small, MixtureFixture, TestFixture},
    StarkEngine,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_bn254_poseidon2::{
            BabyBearBn254Poseidon2Config as RootConfig, BabyBearBn254Poseidon2CpuEngine,
        },
        baby_bear_poseidon2::Digest as InnerDigest,
    },
    utils::setup_tracing,
};
use openvm_static_verifier::{
    field::baby_bear::{BabyBearChip, BabyBearExtChip},
    log_heights_per_air_from_proof, Halo2ProvingMetadata, Halo2ProvingPinning,
    StaticVerifierCircuit, StaticVerifierShape,
};

const MIN_ROWS: usize = 20;

fn select_k_verify_stark(circuit: &StaticVerifierCircuit, proof: &Proof<RootConfig>) -> usize {
    let mut k = 18;
    let mut first_run = true;
    loop {
        let shape = StaticVerifierShape {
            k,
            lookup_bits: k - 1,
            minimum_rows: MIN_ROWS,
            instance_columns: 0,
        };
        let mut builder = StaticVerifierCircuit::builder(CircuitBuilderStage::Keygen, &shape);
        let range = builder.range_chip();
        let ext_chip = BabyBearExtChip::new(BabyBearChip::new(Arc::new(range)));
        let ctx = builder.main(0);
        let _ = circuit.populate_verify_stark_constraints(ctx, &ext_chip, proof);
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
    tracing::info!("Auto-tuned halo2 k={k} (verify-stark constraints only)");
    k
}

#[test]
#[ignore = "too slow"]
fn real_prover_keygen_prove_verify_roundtrip() {
    setup_tracing();
    let system_params = test_system_params_small(2, 8, 3);
    let engine: BabyBearBn254Poseidon2CpuEngine =
        BabyBearBn254Poseidon2CpuEngine::new(system_params);

    let fx = MixtureFixture::standard(5, engine.config().clone());
    let (vk, proof_keygen) = fx.keygen_and_prove(&engine);
    let fx_prove = MixtureFixture::standard(5, engine.config().clone());
    let (_vk_prove, proof_prove) = fx_prove.keygen_and_prove(&engine);

    let log_heights_per_air = log_heights_per_air_from_proof(&proof_keygen);
    let circuit = StaticVerifierCircuit::try_new(vk, InnerDigest::default(), &log_heights_per_air)
        .expect("static circuit params");

    let k = select_k_verify_stark(&circuit, &proof_keygen);
    let shape = StaticVerifierShape {
        k,
        lookup_bits: k - 1,
        minimum_rows: MIN_ROWS,
        instance_columns: 0,
    };
    let params = gen_srs(k as u32);

    // keygen with verify stark constraints only
    let pinning = {
        let mut builder = StaticVerifierCircuit::builder(CircuitBuilderStage::Keygen, &shape);
        let range = builder.range_chip();
        let ext_chip = BabyBearExtChip::new(BabyBearChip::new(Arc::new(range)));
        let ctx = builder.main(0);
        let _proof_wire = circuit.populate_verify_stark_constraints(ctx, &ext_chip, &proof_keygen);

        let config_params = builder.calculate_params(Some(shape.minimum_rows));

        let vk = keygen_vk(&params, &builder).expect("keygen_vk should succeed");
        let pk = keygen_pk(&params, vk, &builder).expect("keygen_pk should succeed");
        let break_points = builder.break_points();

        Halo2ProvingPinning {
            pk,
            metadata: Halo2ProvingMetadata {
                config_params,
                break_points,
                num_pvs: vec![0],
            },
        }
    };
    let halo2_proof =
        circuit.prove_verify_stark_constraints_only(&params, &pinning, &shape, &proof_prove);

    assert!(StaticVerifierCircuit::verify(
        &params,
        pinning.pk.get_vk(),
        &halo2_proof
    ));
}
