use core::cmp::Reverse;

use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
    openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey0, proof::Proof},
};

pub(crate) fn compute_trace_id_to_air_id(
    mvk0: &MultiStarkVerifyingKey0<RootConfig>,
    proof: &Proof<RootConfig>,
) -> Vec<usize> {
    let num_airs = mvk0.per_air.len();
    let mut trace_id_to_air_id: Vec<usize> = (0..num_airs).collect();
    trace_id_to_air_id.sort_by_key(|&air_id| {
        (
            proof.trace_vdata[air_id].is_none(),
            proof.trace_vdata[air_id]
                .as_ref()
                .map(|vdata| Reverse(vdata.log_height)),
            air_id,
        )
    });

    let num_traces = proof.trace_vdata.iter().flatten().count();
    trace_id_to_air_id.truncate(num_traces);
    trace_id_to_air_id
}

/// Permutation of AIR indices when every AIR has a trace, ordered by descending `log_height`
/// (tie-break: lower `air_id` first). Must match [`compute_trace_id_to_air_id`] on such proofs.
pub(crate) fn trace_id_order_from_static_heights(
    mvk0: &MultiStarkVerifyingKey0<RootConfig>,
    log_heights_per_air: &[usize],
) -> Vec<usize> {
    let num_airs = mvk0.per_air.len();
    assert_eq!(
        log_heights_per_air.len(),
        num_airs,
        "log_heights_per_air length must match VK per_air count"
    );
    let mut trace_id_to_air_id: Vec<usize> = (0..num_airs).collect();
    trace_id_to_air_id.sort_by_key(|&air_id| (Reverse(log_heights_per_air[air_id]), air_id));
    trace_id_to_air_id
}
