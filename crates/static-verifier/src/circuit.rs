//! Host-fixed parameters for the static verifier Halo2 circuit (see crate `lib.rs`).

use core::cmp::Reverse;
use std::{borrow::Borrow, fmt, sync::Arc};

use halo2_base::{
    gates::circuit::builder::BaseCircuitBuilder, halo2_proofs::halo2curves::bn256::Fr, Context,
};
use itertools::Itertools;
use openvm_cpu_backend::CpuBackend;
use openvm_recursion_circuit::{
    batch_constraint::expr_eval::DagCommitPvs,
    system::{VerifierConfig, VerifierSubCircuit, VerifierTraceGen},
};
use openvm_stark_sdk::{
    config::{
        baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, Digest as InnerDigest},
    },
    openvm_stark_backend::{
        keygen::types::{MultiStarkVerifyingKey, MultiStarkVerifyingKey0},
        proof::Proof,
        prover::stacked_pcs::StackedLayout,
    },
};
use openvm_verify_stark_host::pvs::CONSTRAINT_EVAL_AIR_ID;

use crate::{
    field::baby_bear::{BabyBearChip, BabyBearExtChip},
    stages::{
        full_pipeline::{
            constrained_verify, extract_public_values, load_proof_wire, ProofWire,
            StaticVerifierPvs,
        },
        proof_shape::trace_id_order_from_static_heights,
    },
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
    pub root_vk: MultiStarkVerifyingKey<RootConfig>,
    /// The [RootConfig] hash onion commitment to the internal-recursive verifier circuit's
    /// symbolic constraints DAG. This is exposed as a public value by the DagCommitSubAir within
    /// the SymbolicExpressionAir in the RootVerifierCircuit.
    ///
    /// This value can be obtained from `cached_trace_record.dag_commit_info.commit` in the
    /// `RootProver`.
    pub internal_recursive_dag_onion_commit: InnerDigest,
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
        root_vk: MultiStarkVerifyingKey<RootConfig>,
        internal_recursive_dag_onion_commit: InnerDigest,
        log_heights_per_air: &[usize],
    ) -> Result<Self, StaticCircuitParamsError> {
        let n = root_vk.inner.per_air.len();
        if log_heights_per_air.len() != n {
            return Err(StaticCircuitParamsError::LogHeightsLenMismatch {
                expected: n,
                got: log_heights_per_air.len(),
            });
        }
        let log_heights_per_air = log_heights_per_air.to_vec();
        let trace_id_to_air_id =
            trace_id_order_from_static_heights(&root_vk.inner, &log_heights_per_air);
        let stacked_layouts =
            build_stacked_layouts_for_static_vk(&root_vk.inner, &log_heights_per_air);
        Ok(Self {
            root_vk,
            internal_recursive_dag_onion_commit,
            log_heights_per_air,
            trace_id_to_air_id,
            stacked_layouts,
        })
    }

    /// STARK verification constraints only: load the proof witness and run
    /// `constrained_verify`.
    ///
    /// Does **not** check proof public values or cached trace commitments, which requires the proof
    /// to have a particular shape following the VM Continuations framework.
    ///
    /// This function should be used internally or for testing only.
    /// Production uses with the continuations framework **must** use [`Self::populate`] instead.
    pub fn populate_verify_stark_constraints(
        &self,
        ctx: &mut Context<Fr>,
        ext_chip: &BabyBearExtChip,
        proof: &Proof<RootConfig>,
    ) -> ProofWire {
        let mut profiler = crate::profiling::CellProfiler::new("static_verifier", ctx.advice.len());

        profiler.push("load_proof_wire", ctx.advice.len());
        let proof_wire = load_proof_wire(ctx, ext_chip, proof, &self.log_heights_per_air);
        profiler.pop(ctx.advice.len());

        profiler.push("constrained_verify", ctx.advice.len());
        constrained_verify(
            ctx,
            ext_chip,
            &self.root_vk,
            &proof_wire,
            &self.trace_id_to_air_id,
            &self.log_heights_per_air,
            &self.stacked_layouts,
        );
        profiler.pop(ctx.advice.len());

        profiler.print(ctx.advice.len());

        #[cfg(feature = "cell-profiling")]
        if let Ok(dir) = std::env::var("OPENVM_PROFILE_DIR") {
            let _ = std::fs::create_dir_all(&dir);
            profiler.write_flamegraph(
                &format!("{dir}/static_verifier_constraints.svg"),
                "Static Verifier Constraints",
                ctx.advice.len(),
            );
            profiler.write_flamegraph_reversed(
                &format!("{dir}/static_verifier_constraints_rev.svg"),
                "Static Verifier Constraints (reversed)",
                ctx.advice.len(),
            );
        }

        proof_wire
    }

    /// Populate a builder with the static verifier constraints and return the public values.
    pub fn populate(
        &self,
        builder: &mut BaseCircuitBuilder<Fr>,
        proof: &Proof<RootConfig>,
    ) -> StaticVerifierPvs<Fr> {
        let range = builder.range_chip();
        let ext_chip = BabyBearExtChip::new(BabyBearChip::new(Arc::new(range)));
        let ctx = builder.main(0);

        let mut profiler = crate::profiling::CellProfiler::new("populate", ctx.advice.len());

        profiler.push("verify_stark_constraints", ctx.advice.len());
        let proof_wire = &self.populate_verify_stark_constraints(ctx, &ext_chip, proof);
        profiler.pop(ctx.advice.len());

        debug_assert!(
            proof_wire
                .cached_commitment_roots
                .iter()
                .all(|commits| commits.is_empty()),
            "RootVerifierCircuit has no cached trace"
        );
        profiler.push("pin_dag_onion_commit", ctx.advice.len());
        let &DagCommitPvs::<_> {
            commit: onion_commit,
        } = proof_wire.public_values[CONSTRAINT_EVAL_AIR_ID]
            .as_slice()
            .borrow();
        for (bb_wire, bb_const) in onion_commit
            .into_iter()
            .zip_eq(self.internal_recursive_dag_onion_commit)
        {
            let loaded_const = ext_chip.base().load_constant(ctx, bb_const);
            ext_chip.base().assert_equal(ctx, bb_wire, loaded_const);
        }
        profiler.pop(ctx.advice.len());

        profiler.push("extract_public_values", ctx.advice.len());
        let pvs_wire = extract_public_values(ctx, ext_chip.base(), proof_wire);
        profiler.pop(ctx.advice.len());

        #[cfg(feature = "cell-profiling")]
        if let Ok(dir) = std::env::var("OPENVM_PROFILE_DIR") {
            let _ = std::fs::create_dir_all(&dir);
            profiler.write_flamegraph(
                &format!("{dir}/populate.svg"),
                "Static Verifier Populate",
                ctx.advice.len(),
            );
            profiler.write_flamegraph_reversed(
                &format!("{dir}/populate_rev.svg"),
                "Static Verifier Populate (reversed)",
                ctx.advice.len(),
            );
        }

        let pvs_vec = pvs_wire.to_vec();
        let pvs_fr = pvs_vec.iter().map(|v| *v.value()).collect_vec();
        builder.assigned_instances[0].extend(pvs_vec);

        StaticVerifierPvs::from_slice(&pvs_fr)
    }
}

pub fn compute_dag_onion_commit(
    internal_recursive_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
) -> InnerDigest {
    // Note: the MAX_NUM_PROOFS const generic does not impact the build_cached_trace_record function
    // used internally below, but we use 1 to match the root circuit. The internal_recursive circuit
    // itself uses MAX_NUM_PROOFS = 3, but here it is the child.
    let verifier_circuit = VerifierSubCircuit::<1>::new_with_options(
        Arc::new(internal_recursive_vk.clone()),
        VerifierConfig {
            continuations_enabled: true,
            has_cached: false,
            ..Default::default()
        },
    );

    // The ProverBackend and config used here also does not impact the generated CachedTraceRecord
    let cached_trace_record = VerifierTraceGen::<
        CpuBackend<BabyBearPoseidon2Config>,
        BabyBearPoseidon2Config,
    >::cached_trace_record(&verifier_circuit, internal_recursive_vk);

    // Despite the name, this returns the DAG onion commit for when there is no cached trace
    cached_trace_record.dag_commit_info.unwrap().commit
}
