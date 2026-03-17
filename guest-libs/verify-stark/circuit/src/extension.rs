use std::borrow::Borrow;

use eyre::Result;
use itertools::Itertools;
use openvm_circuit::{
    arch::{
        deferral::{DeferralState, OutputRaw},
        hasher::poseidon2::vm_poseidon2_hasher,
    },
    system::program::trace::compute_exe_commit,
};
use openvm_continuations::{
    circuit::utils::vk_commit_components, utils::poseidon2_input_to_digests,
};
use openvm_deferral_circuit::{
    generate_deferral_results, poseidon2::deferral_poseidon2_chip, RawDeferralResult,
};
use openvm_recursion_circuit::utils::poseidon2_hash_slice;
use openvm_stark_backend::{
    codec::Decode, p3_field::PrimeField32, verifier::verify, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, poseidon2_compress_with_capacity,
    BabyBearPoseidon2Config as SC, F,
};
use openvm_verify_stark_host::{
    pvs::{VerifierBasePvs, VmPvs, VERIFIER_PVS_AIR_ID, VM_PVS_AIR_ID},
    verify_vm_stark_proof_pvs,
    vk::NonRootStarkVerifyingKey,
    NonRootStarkProof,
};

///////////////////////////////////////////////////////////////////////////////
/// DEFERRAL FN IMPLEMENTATION
///////////////////////////////////////////////////////////////////////////////
pub fn verify_stark_deferral_fn(encoded_proof: &[u8]) -> OutputRaw {
    let proof = NonRootStarkProof::decode_from_bytes(encoded_proof).unwrap();
    output_raw_from_proof(&proof)
}

fn output_raw_from_proof(proof: &NonRootStarkProof) -> OutputRaw {
    // get (app_exe_commit, app_vk_commit, public values)
    let (base_pvs_slice, _) = proof.inner.public_values[VERIFIER_PVS_AIR_ID]
        .as_slice()
        .split_at(VerifierBasePvs::<u8>::width());
    let verifier_pvs: &VerifierBasePvs<F> = base_pvs_slice.borrow();
    let vm_pvs: &VmPvs<F> = proof.inner.public_values[VM_PVS_AIR_ID].as_slice().borrow();

    let app_exe_commit = compute_exe_commit(
        &vm_poseidon2_hasher(),
        &vm_pvs.program_commit,
        &vm_pvs.initial_root,
        vm_pvs.initial_pc,
    );
    let app_vk_commit =
        poseidon2_hash_slice(&vk_commit_components(verifier_pvs).into_flattened()).0;

    let output_f = app_exe_commit
        .into_iter()
        .chain(app_vk_commit)
        .chain(proof.user_pvs_proof.public_values.iter().copied())
        .collect_vec();
    f_slice_to_bytes(&output_f)
}

fn f_slice_to_bytes(slice: &[F]) -> Vec<u8> {
    let mut output = Vec::with_capacity(size_of_val(slice));
    for value in slice {
        let bytes = value.as_canonical_u32().to_le_bytes();
        output.extend_from_slice(&bytes);
    }
    output
}

///////////////////////////////////////////////////////////////////////////////
/// DEFERRAL STATE GENERATION
///////////////////////////////////////////////////////////////////////////////
pub fn get_raw_deferral_results(
    vk: &NonRootStarkVerifyingKey,
    proofs: &[NonRootStarkProof],
) -> Result<Vec<RawDeferralResult>> {
    let config = SC::default_from_params(vk.mvk.inner.params.clone());

    proofs
        .iter()
        .map(|proof| {
            verify_vm_stark_proof_pvs(vk, proof)?;

            let mut ts = default_duplex_sponge_recorder();
            verify(&config, &vk.mvk, &proof.inner, &mut ts)?;

            let final_ts_state = *ts.into_log().perm_results().last().unwrap();
            let (left_ts, right_ts) = poseidon2_input_to_digests(final_ts_state);
            let ts_commit = poseidon2_compress_with_capacity(left_ts, right_ts).0;
            let cached_commit = vk.baseline.internal_recursive_dag_commit.cached_commit;
            // let input_commit =
            //     poseidon2_hash_slice(&vec![ts_commit, cached_commit].into_flattened()).0;

            // TODO[INT-6415]: hash slice, not compress
            let input_commit = poseidon2_compress_with_capacity(ts_commit, cached_commit).0;

            Ok(RawDeferralResult {
                input: f_slice_to_bytes(&input_commit),
                output_raw: output_raw_from_proof(proof),
            })
        })
        .collect()
}

pub fn get_deferral_state(
    vk: &NonRootStarkVerifyingKey,
    proofs: &[NonRootStarkProof],
    deferral_idx: u32,
) -> Result<DeferralState> {
    let raw_results = get_raw_deferral_results(vk, proofs)?;
    let results =
        generate_deferral_results(raw_results, deferral_idx, &deferral_poseidon2_chip::<F>());
    Ok(DeferralState::new(results))
}
