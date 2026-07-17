//! Integration: one Halo2 KZG roundtrip on a BN254 [`MixtureFixture`] proof using only
//! [`StaticVerifierCircuit::populate_verify_stark_constraints`] (no continuations public values or
//! DAG cached-commit pin).
//!
//! Full [`StaticVerifierCircuit::populate`] end-to-end is exercised in `openvm-sdk` integration
//! tests, not here.

use std::sync::Arc;

use halo2_base::{
    gates::circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage},
    halo2_proofs::{
        halo2curves::bn256::{Bn256, G1Affine},
        plonk::{create_proof, keygen_pk, keygen_vk},
        poly::kzg::{commitment::KZGCommitmentScheme, multiopen::ProverSHPLONK},
        transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
    },
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
    halo2_backend::Halo2Backend, log_heights_per_air_from_proof, Fr, Halo2Params,
    Halo2ProvingMetadata, Halo2ProvingPinning, StaticVerifierCircuit, StaticVerifierProof,
    StaticVerifierShape,
};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

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
        let ctx = builder.main(0);
        let mut backend = Halo2Backend::new(Arc::new(range), ctx);
        let _ = circuit.populate_verify_stark_constraints(&mut backend, proof);
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

fn prove_verify_stark_constraints_only(
    circuit: &StaticVerifierCircuit,
    params: &Halo2Params,
    pinning: &Halo2ProvingPinning,
    proof: &Proof<RootConfig>,
) -> StaticVerifierProof {
    let mut builder = BaseCircuitBuilder::prover(
        pinning.metadata.config_params.clone(),
        pinning.metadata.break_points.clone(),
    );
    builder = builder.use_instance_columns(0);

    let range = builder.range_chip();
    let ctx = builder.main(0);
    let mut backend = Halo2Backend::new(Arc::new(range), ctx);
    let _ = circuit.populate_verify_stark_constraints(&mut backend, proof);

    let rng = ChaCha20Rng::from_seed(Default::default());
    let instances: &[&[Fr]] = &[];
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    create_proof::<
        KZGCommitmentScheme<Bn256>,
        ProverSHPLONK<'_, Bn256>,
        Challenge255<_>,
        _,
        Blake2bWrite<Vec<u8>, G1Affine, _>,
        _,
    >(
        params,
        &pinning.pk,
        &[builder],
        &[instances],
        rng,
        &mut transcript,
    )
    .expect("Halo2 proof generation should succeed");

    StaticVerifierProof {
        proof_bytes: transcript.finalize(),
        public_inputs: Vec::new(),
    }
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
        let ctx = builder.main(0);
        let mut backend = Halo2Backend::new(Arc::new(range), ctx);
        let _proof_wire = circuit.populate_verify_stark_constraints(&mut backend, &proof_keygen);

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
    assert_eq!(shape.instance_columns, 0);
    let halo2_proof =
        prove_verify_stark_constraints_only(&circuit, &params, &pinning, &proof_prove);

    assert!(StaticVerifierCircuit::verify(
        &params,
        pinning.pk.get_vk(),
        &halo2_proof
    ));
}
