use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        BabyBearBn254Poseidon2Config as NativeConfig, Digest as NativeDigest, EF as NativeEF,
        F as NativeF,
    },
    openvm_stark_backend::{
        FiatShamirTranscript, StarkProtocolConfig,
        keygen::types::{MultiStarkVerifyingKey, MultiStarkVerifyingKey0},
        p3_field::{PrimeCharacteristicRing, TwoAdicField},
        poly_common::Squarable,
        proof::Proof,
        prover::stacked_pcs::StackedLayout,
    },
};

use super::{
    batch_constraints::{
        BatchConstraintError, BatchIntermediates, coeffs_to_native_ext as batch_coeffs_to_ext,
        compute_trace_id_to_air_id, derive_batch_intermediates_with_inputs,
        enforce_trace_height_constraints, observe_preamble,
    },
    proof_shape::derive_proof_shape_rules,
};

#[derive(Clone, Debug)]
pub(crate) struct PreparedPipelineInputs {
    pub trace_id_to_air_id: Vec<usize>,
    pub layouts: Vec<StackedLayout>,
    pub l_skip: usize,
    pub omega_skip_pows: Vec<NativeF>,
    pub batch: BatchIntermediates,
    pub r: Vec<NativeEF>,
    pub need_rot_per_commit: Vec<Vec<bool>>,
}

pub(crate) fn trace_log_height(
    proof: &Proof<NativeConfig>,
    air_id: usize,
) -> Result<usize, BatchConstraintError> {
    proof.trace_vdata[air_id]
        .as_ref()
        .map(|vdata| vdata.log_height)
        .ok_or(BatchConstraintError::MissingTraceVData { air_id })
}

pub(crate) fn derive_n_per_trace(
    proof: &Proof<NativeConfig>,
    trace_id_to_air_id: &[usize],
    l_skip: usize,
) -> Result<Vec<isize>, BatchConstraintError> {
    trace_id_to_air_id
        .iter()
        .map(|&air_id| Ok(trace_log_height(proof, air_id)? as isize - l_skip as isize))
        .collect::<Result<Vec<_>, _>>()
}

pub(crate) fn derive_need_rot_per_commit(
    mvk0: &MultiStarkVerifyingKey0<NativeConfig>,
    proof: &Proof<NativeConfig>,
    trace_id_to_air_id: &[usize],
) -> Result<Vec<Vec<bool>>, BatchConstraintError> {
    let mut need_rot_per_commit = vec![
        trace_id_to_air_id
            .iter()
            .map(|&air_id| mvk0.per_air[air_id].params.need_rot)
            .collect::<Vec<_>>(),
    ];
    for &air_id in trace_id_to_air_id {
        let need_rot = mvk0.per_air[air_id].params.need_rot;
        if mvk0.per_air[air_id].preprocessed_data.is_some() {
            need_rot_per_commit.push(vec![need_rot]);
        }
        let cached_len = proof.trace_vdata[air_id]
            .as_ref()
            .ok_or(BatchConstraintError::MissingTraceVData { air_id })?
            .cached_commitments
            .len();
        for _ in 0..cached_len {
            need_rot_per_commit.push(vec![need_rot]);
        }
    }
    Ok(need_rot_per_commit)
}

pub(crate) fn collect_trace_commitments(
    mvk0: &MultiStarkVerifyingKey0<NativeConfig>,
    proof: &Proof<NativeConfig>,
    trace_id_to_air_id: &[usize],
) -> Result<Vec<NativeDigest>, BatchConstraintError> {
    let mut commits = vec![proof.common_main_commit];
    for &air_id in trace_id_to_air_id {
        if let Some(preprocessed) = &mvk0.per_air[air_id].preprocessed_data {
            commits.push(preprocessed.commit);
        }
        commits.extend(
            &proof.trace_vdata[air_id]
                .as_ref()
                .ok_or(BatchConstraintError::MissingTraceVData { air_id })?
                .cached_commitments,
        );
    }
    Ok(commits)
}

pub(crate) fn derive_u_cube_from_prism(
    u_prism: &[NativeEF],
    l_skip: usize,
) -> Result<Vec<NativeEF>, BatchConstraintError> {
    let (&u0, u_rest) = u_prism
        .split_first()
        .ok_or(BatchConstraintError::MissingStackedChallenges)?;
    Ok(u0
        .exp_powers_of_2()
        .take(l_skip)
        .chain(u_rest.iter().copied())
        .collect::<Vec<_>>())
}

pub(crate) fn prepare_pipeline_inputs<TS: FiatShamirTranscript<NativeConfig>>(
    transcript: &mut TS,
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<PreparedPipelineInputs, BatchConstraintError> {
    if config.params() != &mvk.inner.params {
        return Err(BatchConstraintError::SystemParamsMismatch);
    }

    let mvk0 = &mvk.inner;
    let trace_id_to_air_id = compute_trace_id_to_air_id(mvk0, proof);
    enforce_trace_height_constraints(mvk0, proof, &trace_id_to_air_id)?;

    observe_preamble(transcript, mvk0, mvk.pre_hash, proof);
    let layouts = derive_proof_shape_rules(mvk0, proof)?.layouts;

    let l_skip = mvk0.params.l_skip;
    let n_per_trace = derive_n_per_trace(proof, &trace_id_to_air_id, l_skip)?;

    let omega_skip = NativeF::two_adic_generator(l_skip);
    let omega_skip_pows: Vec<_> = omega_skip.powers().take(1 << l_skip).collect();

    let batch = derive_batch_intermediates_with_inputs(
        transcript,
        mvk0,
        &proof.public_values,
        &proof.gkr_proof,
        &proof.batch_constraint_proof,
        &trace_id_to_air_id,
        &n_per_trace,
        &omega_skip_pows,
    )?;
    let r = batch
        .r
        .iter()
        .copied()
        .map(batch_coeffs_to_ext)
        .collect::<Vec<_>>();
    let need_rot_per_commit = derive_need_rot_per_commit(mvk0, proof, &trace_id_to_air_id)?;

    Ok(PreparedPipelineInputs {
        trace_id_to_air_id,
        layouts,
        l_skip,
        omega_skip_pows,
        batch,
        r,
        need_rot_per_commit,
    })
}
