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
    proof_shape::{compute_trace_id_to_air_id, trace_id_order_from_static_heights},
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

/// Error building [`StaticVerifierCircuit`] from a template child proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaticCircuitParamsError {
    TraceVdataLenMismatch {
        expected: usize,
        got: usize,
    },
    MissingTraceVData {
        air_id: usize,
    },
    /// `compute_trace_id_to_air_id` disagrees with height-only ordering (should not happen for a
    /// valid proof whose `trace_vdata` matches the extracted heights).
    TraceOrderingMismatch,
}

impl fmt::Display for StaticCircuitParamsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TraceVdataLenMismatch { expected, got } => {
                write!(
                    f,
                    "trace_vdata length {got} != VK per_air length {expected}"
                )
            }
            Self::MissingTraceVData { air_id } => {
                write!(f, "missing trace_vdata for air_id {air_id}")
            }
            Self::TraceOrderingMismatch => {
                write!(
                    f,
                    "trace_id_to_air_id from proof does not match static height ordering"
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
    /// Build static parameters from a child VK and any valid root proof with the target shape.
    ///
    /// Every AIR must have `trace_vdata` present. The stored permutation matches
    /// [`compute_trace_id_to_air_id`] on the template proof.
    pub fn try_new(
        child_vk: MultiStarkVerifyingKey<RootConfig>,
        template_proof: &Proof<RootConfig>,
    ) -> Result<Self, StaticCircuitParamsError> {
        let n = child_vk.inner.per_air.len();
        if template_proof.trace_vdata.len() != n {
            return Err(StaticCircuitParamsError::TraceVdataLenMismatch {
                expected: n,
                got: template_proof.trace_vdata.len(),
            });
        }
        let mut log_heights_per_air = Vec::with_capacity(n);
        for (air_id, tv) in template_proof.trace_vdata.iter().enumerate() {
            let Some(vd) = tv.as_ref() else {
                return Err(StaticCircuitParamsError::MissingTraceVData { air_id });
            };
            log_heights_per_air.push(vd.log_height);
        }
        let from_proof = compute_trace_id_to_air_id(&child_vk.inner, template_proof);
        let from_heights =
            trace_id_order_from_static_heights(&child_vk.inner, &log_heights_per_air);
        if from_proof != from_heights {
            return Err(StaticCircuitParamsError::TraceOrderingMismatch);
        }
        let stacked_layouts =
            build_stacked_layouts_for_static_vk(&child_vk.inner, &log_heights_per_air);
        Ok(Self {
            child_vk,
            log_heights_per_air,
            trace_id_to_air_id: from_heights,
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
