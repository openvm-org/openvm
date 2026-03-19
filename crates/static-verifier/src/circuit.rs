//! Host-fixed parameters for the static verifier Halo2 circuit (see crate `lib.rs`).

use core::cmp::Reverse;
use std::fmt;

use halo2_base::{
    gates::circuit::builder::BaseCircuitBuilder, halo2_proofs::halo2curves::bn256::Fr,
};
use itertools::Itertools;
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
    openvm_stark_backend::{
        keygen::types::{MultiStarkVerifyingKey, MultiStarkVerifyingKey0},
        proof::Proof,
        prover::stacked_pcs::StackedLayout,
    },
};

use crate::stages::{
    full_pipeline::{constrained_verify, load_proof_wire},
    proof_shape::trace_id_order_from_static_heights,
};

/// Builds stacked PCS layouts for the static verifier from VK widths and fixed per-air log heights.
pub(crate) fn build_stacked_layouts_for_static_vk(
    mvk0: &MultiStarkVerifyingKey0<RootConfig>,
    log_heights_per_air: &[usize],
) -> Vec<StackedLayout> {
    let l_skip = mvk0.params.l_skip;
    assert_eq!(
        log_heights_per_air.len(),
        mvk0.per_air.len(),
        "log_heights_per_air length must match VK per_air count"
    );
    let mut per_trace = mvk0
        .per_air
        .iter()
        .enumerate()
        .map(|(air_idx, vk)| (air_idx, vk, log_heights_per_air[air_idx]))
        .collect::<Vec<_>>();
    per_trace.sort_by_key(|(_, _, log_height)| Reverse(*log_height));

    let common_main_layout = StackedLayout::new(
        l_skip,
        mvk0.params.n_stack + l_skip,
        per_trace
            .iter()
            .map(|(_, vk, log_height)| (vk.params.width.common_main, *log_height))
            .collect::<Vec<_>>(),
    )
    .expect("stacked layout for common main");
    let other_layouts = per_trace
        .iter()
        .flat_map(|(_, vk, log_height)| {
            vk.params
                .width
                .preprocessed
                .iter()
                .chain(&vk.params.width.cached_mains)
                .copied()
                .map(|width| (width, *log_height))
                .collect::<Vec<_>>()
        })
        .map(|sorted| {
            StackedLayout::new(l_skip, mvk0.params.n_stack + l_skip, vec![sorted])
                .expect("stacked layout for auxiliary column")
        })
        .collect::<Vec<_>>();
    core::iter::once(common_main_layout)
        .chain(other_layouts)
        .collect::<Vec<_>>()
}

/// Error building [`StaticVerifierCircuit`] from fixed per-AIR log heights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaticCircuitParamsError {
    LogHeightsLenMismatch { expected: usize, got: usize },
}

impl fmt::Display for StaticCircuitParamsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LogHeightsLenMismatch { expected, got } => {
                write!(
                    f,
                    "log_heights_per_air length {got} != VK per_air length {expected}"
                )
            }
        }
    }
}

impl std::error::Error for StaticCircuitParamsError {}

/// Parameters fixed host-side for the static verifier (child VK, trace heights, AIR permutation).
#[derive(Clone, Debug)]
pub struct StaticVerifierCircuit {
    pub child_vk: MultiStarkVerifyingKey<RootConfig>,
    pub log_heights_per_air: Vec<usize>,
    pub trace_id_to_air_id: Vec<usize>,
    pub stacked_layouts: Vec<StackedLayout>,
}

impl StaticVerifierCircuit {
    /// Build static parameters from a child VK and the per-AIR trace log heights for this circuit.
    ///
    /// `log_heights_per_air[i]` is the log₂ trace height for AIR `i` (same indexing as the child
    /// VK's `per_air`). Trace IDs are ordered by descending height (tie-break: lower `air_id`
    /// first).
    pub fn try_new(
        child_vk: MultiStarkVerifyingKey<RootConfig>,
        log_heights_per_air: &[usize],
    ) -> Result<Self, StaticCircuitParamsError> {
        let n = child_vk.inner.per_air.len();
        if log_heights_per_air.len() != n {
            return Err(StaticCircuitParamsError::LogHeightsLenMismatch {
                expected: n,
                got: log_heights_per_air.len(),
            });
        }
        let log_heights_per_air = log_heights_per_air.to_vec();
        let trace_id_to_air_id =
            trace_id_order_from_static_heights(&child_vk.inner, &log_heights_per_air);
        let stacked_layouts =
            build_stacked_layouts_for_static_vk(&child_vk.inner, &log_heights_per_air);
        Ok(Self {
            child_vk,
            log_heights_per_air,
            trace_id_to_air_id,
            stacked_layouts,
        })
    }

    /// Populate a builder with the static verifier constraints and return the public inputs.
    pub fn populate(
        &self,
        builder: &mut BaseCircuitBuilder<Fr>,
        proof: &Proof<RootConfig>,
    ) -> Vec<Fr> {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let proof_wire = load_proof_wire(ctx, &range, proof, &self.log_heights_per_air);
        let statement_public_inputs = constrained_verify(
            ctx,
            &range,
            &self.child_vk,
            proof_wire,
            &self.trace_id_to_air_id,
            &self.log_heights_per_air,
            &self.stacked_layouts,
        );
        let pis = statement_public_inputs
            .iter()
            .map(|c| *c.value())
            .collect_vec();
        builder.assigned_instances[0].extend(statement_public_inputs);
        pis
    }
}
