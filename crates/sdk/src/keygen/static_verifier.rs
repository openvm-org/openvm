use std::sync::Arc;

use openvm_continuations::{CommitBytes, RootSC};
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, proof::Proof};
use openvm_static_verifier::{
    log_heights_per_air_from_proof, Halo2Params, Halo2ParamsReader, Halo2WrapperProvingKey,
    StaticVerifierCircuit, StaticVerifierProvingKey, StaticVerifierShape,
};

use crate::{config::Halo2Config, keygen::Halo2ProvingKey};

/// Generate a [`Halo2ProvingKey`] (static verifier + wrapper) by running a
/// dummy root proof through the pipeline.
///
/// This is the self-contained keygen flow:
/// 1. Build a [`StaticVerifierProvingKey`] from the root VK and proof shape
/// 2. Generate a dummy snark from the static verifier
/// 3. Build a [`Halo2WrapperProvingKey`] (auto-tuned or fixed `k`)
/// 4. Return the composite [`Halo2ProvingKey`]
pub fn keygen_halo2(
    halo2_config: &Halo2Config,
    reader: &impl Halo2ParamsReader,
    shape: StaticVerifierShape,
    root_vk: &MultiStarkVerifyingKey<RootSC>,
    internal_recursive_dag_cached_commit: CommitBytes,
    dummy_root_proof: &Proof<RootSC>,
) -> Halo2ProvingKey {
    let params = reader.read_params(halo2_config.verifier_k);

    let verifier = keygen_static_verifier(
        &params,
        shape,
        root_vk,
        internal_recursive_dag_cached_commit,
        dummy_root_proof,
    );

    let dummy_snark = verifier.generate_dummy_snark(reader);

    let wrapper = if let Some(wrapper_k) = halo2_config.wrapper_k {
        Halo2WrapperProvingKey::keygen(&reader.read_params(wrapper_k), dummy_snark)
    } else {
        Halo2WrapperProvingKey::keygen_auto_tune(reader, dummy_snark)
    };

    Halo2ProvingKey {
        verifier: Arc::new(verifier),
        wrapper: Arc::new(wrapper),
        profiling: halo2_config.profiling,
    }
}

/// Generate a [`StaticVerifierProvingKey`] from a root VK, heights, and a
/// dummy root proof. This is the lower-level keygen without the wrapper.
pub fn keygen_static_verifier(
    params: &Halo2Params,
    shape: StaticVerifierShape,
    root_vk: &MultiStarkVerifyingKey<RootSC>,
    internal_recursive_dag_cached_commit: CommitBytes,
    dummy_root_proof: &Proof<RootSC>,
) -> StaticVerifierProvingKey {
    let log_heights = log_heights_per_air_from_proof(dummy_root_proof);

    // Convert CommitBytes → RootDigest ([Bn254; 1])
    use p3_bn254::Bn254;
    let bn254: Bn254 = internal_recursive_dag_cached_commit.into();
    let root_digest = [bn254];

    let circuit = StaticVerifierCircuit::try_new(root_vk.clone(), root_digest, &log_heights)
        .expect("Failed to construct StaticVerifierCircuit");

    StaticVerifierProvingKey::keygen(params, shape, circuit, dummy_root_proof)
}
