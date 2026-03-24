use std::sync::Arc;

use openvm_continuations::{RootSC, SC};
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, proof::Proof};
use openvm_static_verifier::{
    compute_dag_onion_commit, log_heights_per_air_from_proof, Halo2Params, Halo2ParamsReader,
    Halo2WrapperProvingKey, StaticVerifierCircuit, StaticVerifierProvingKey, StaticVerifierShape,
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
    internal_recursive_vk: &MultiStarkVerifyingKey<SC>,
    root_vk: &MultiStarkVerifyingKey<RootSC>,
    dummy_root_proof: &Proof<RootSC>,
) -> Halo2ProvingKey {
    let params = reader.read_params(halo2_config.verifier_k);

    let verifier = keygen_static_verifier(
        &params,
        shape,
        internal_recursive_vk,
        root_vk,
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
    internal_recursive_vk: &MultiStarkVerifyingKey<SC>,
    root_vk: &MultiStarkVerifyingKey<RootSC>,
    dummy_root_proof: &Proof<RootSC>,
) -> StaticVerifierProvingKey {
    let log_heights = log_heights_per_air_from_proof(dummy_root_proof);
    let onion_commit = compute_dag_onion_commit(internal_recursive_vk);

    let circuit = StaticVerifierCircuit::try_new(root_vk.clone(), onion_commit, &log_heights)
        .expect("Failed to construct StaticVerifierCircuit");

    StaticVerifierProvingKey::keygen(params, shape, circuit, dummy_root_proof)
}
