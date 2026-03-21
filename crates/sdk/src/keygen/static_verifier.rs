use openvm_continuations::{CommitBytes, RootSC};
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, proof::Proof};
use openvm_static_verifier::{
    log_heights_per_air_from_proof, Halo2Params, StaticVerifierCircuit, StaticVerifierProvingKey,
    StaticVerifierShape,
};

/// Generate a [`StaticVerifierProvingKey`] by running a dummy root proof through the pipeline.
///
/// This is the self-contained keygen flow:
/// 1. Use a pre-generated dummy root proof
/// 2. Extract per-AIR log heights from the dummy proof
/// 3. Build a [`StaticVerifierCircuit`] from the root VK and heights
/// 4. Run keygen to produce the final Halo2 proving key
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
