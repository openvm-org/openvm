use core::cmp::Reverse;

use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
    openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey0, proof::Proof},
};

/// Per-AIR log₂ trace heights from `proof.trace_vdata`, in AIR index order.
///
/// Panics if any entry is [`None`]. The static verifier requires a trace for every AIR.
pub fn log_heights_per_air_from_proof(proof: &Proof<RootConfig>) -> Vec<usize> {
    proof
        .trace_vdata
        .iter()
        .enumerate()
        .map(|(air_id, tv)| {
            tv.as_ref()
                .unwrap_or_else(|| panic!("missing trace_vdata for air_id {air_id}"))
                .log_height
        })
        .collect()
}

/// Permutation of AIR indices when every AIR has a trace, ordered by descending `log_height`
/// (tie-break: lower `air_id` first). For a proof with full `trace_vdata`, this matches that
/// proof's trace ordering.
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
