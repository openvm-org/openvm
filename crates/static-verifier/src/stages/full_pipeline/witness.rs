use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
    openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey0, proof::Proof},
};

use crate::stages::batch_constraints::BatchConstraintError;

/// Helper function, purely on out-of-circuit values. `builder` is not involved and there are no
/// cells.
pub(crate) fn get_need_rot_per_commit(
    mvk0: &MultiStarkVerifyingKey0<RootConfig>,
    proof: &Proof<RootConfig>,
    trace_id_to_air_id: &[usize],
) -> Result<Vec<Vec<bool>>, BatchConstraintError> {
    let mut need_rot_per_commit = vec![trace_id_to_air_id
        .iter()
        .map(|&air_id| mvk0.per_air[air_id].params.need_rot)
        .collect::<Vec<_>>()];
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
