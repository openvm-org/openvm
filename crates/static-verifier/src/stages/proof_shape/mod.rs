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
