use std::sync::Arc;

use eyre::Result;
use openvm_continuations::{
    bn254::CommitBytes,
    prover::{DeferralChildVkKind, DeferralCircuitProver},
    SC,
};
use openvm_deferral_circuit::{DeferralExtension, DeferralFn};
use openvm_recursion_circuit::utils::poseidon2_hash_slice;
use openvm_stark_backend::{
    keygen::types::MultiStarkProvingKey,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    proof::Proof,
    SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{poseidon2_compress_with_capacity, F};
use openvm_verify_stark_host::pvs::DeferralPvs;
use tracing::info_span;

use crate::{config::AggregationConfig, keygen::AggProvingKey, DeferralInput};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_continuations::prover::DeferralInnerGpuProver as DeferralInnerProver;
        use openvm_continuations::prover::DeferralHookGpuProver as DeferralHookProver;
        type E = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
    } else {
        use openvm_continuations::prover::DeferralInnerCpuProver as DeferralInnerProver;
        use openvm_continuations::prover::DeferralHookCpuProver as DeferralHookProver;
        type E = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
    }
}

mod circuit;
mod merkle;
pub use circuit::*;
pub use merkle::*;

pub type DefAggProvingKey = AggProvingKey;

#[allow(clippy::large_enum_variant)]
pub enum DeferralProof {
    Present(Proof<SC>),
    Absent(DeferralPvs<F>),
}

pub struct DeferralProver {
    pub single_circuit_provers: Vec<SingleDefCircuitProver>,
    pub internal_recursive_prover: DeferralInnerProver,
    pub def_hook_prover: DeferralHookProver,
}

unsafe impl Send for DeferralProver {}
unsafe impl Sync for DeferralProver {}

impl DeferralProver {
    pub fn new<DP: DeferralCircuitProver<SC> + 'static>(
        def_circuit_prover: DP,
        agg_config: AggregationConfig,
        hook_params: SystemParams,
    ) -> Self {
        let single_circuit_prover = SingleDefCircuitProver::new(
            def_circuit_prover,
            agg_config.params.leaf,
            agg_config.params.internal.clone(),
        );
        let internal_recursive_prover = DeferralInnerProver::new::<E>(
            single_circuit_prover.internal_for_leaf_prover.get_vk(),
            agg_config.params.internal,
            true,
        );
        let def_hook_prover =
            DeferralHookProver::new::<E>(internal_recursive_prover.get_vk(), hook_params);
        Self {
            single_circuit_provers: vec![single_circuit_prover],
            internal_recursive_prover,
            def_hook_prover,
        }
    }

    pub fn from_pks<DP: DeferralCircuitProver<SC> + 'static>(
        def_circuit_prover: DP,
        def_agg_pk: DefAggProvingKey,
        def_hook_pk: Arc<MultiStarkProvingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
    ) -> Self {
        let single_circuit_prover = SingleDefCircuitProver::from_pks(
            def_circuit_prover,
            def_agg_pk.leaf_pk,
            def_agg_pk.internal_for_leaf_pk,
        );
        let internal_recursive_prover = DeferralInnerProver::from_pk::<E>(
            single_circuit_prover.internal_for_leaf_prover.get_vk(),
            def_agg_pk.internal_recursive_pk,
            false,
        );
        let def_hook_prover = DeferralHookProver::from_pk::<E>(
            internal_recursive_prover.get_vk(),
            internal_recursive_cached_commit,
            def_hook_pk,
        );
        Self {
            single_circuit_provers: vec![single_circuit_prover],
            internal_recursive_prover,
            def_hook_prover,
        }
    }

    pub fn with_prover<DP: DeferralCircuitProver<SC> + 'static>(
        mut self,
        def_circuit_prover: DP,
    ) -> Self {
        let leaf_params = self.single_circuit_provers[0]
            .leaf_prover
            .get_vk()
            .inner
            .params
            .clone();
        let internal_params = self.internal_recursive_prover.get_vk().inner.params.clone();
        let single_circuit_prover =
            SingleDefCircuitProver::new(def_circuit_prover, leaf_params, internal_params);
        self.single_circuit_provers.push(single_circuit_prover);
        self
    }

    pub fn prove(&self, inputs: &[DeferralInput]) -> Result<Vec<DeferralProof>> {
        // Generate internal-for-leaf proofs and leaf IO commits per circuit
        let per_circuit = self
            .single_circuit_provers
            .iter()
            .zip(inputs)
            .map(|(prover, inputs)| prover.prove(inputs))
            .collect::<Result<Vec<_>>>()?;

        // For each circuit: do internal recursive aggregation then generate the hook proof
        per_circuit
            .into_iter()
            .enumerate()
            .map(|(circuit_idx, res)| {
                let mut proofs = res.internal_for_leaf_proofs;
                if proofs.is_empty() {
                    let def_vk_commit = self.single_circuit_provers[circuit_idx]
                        .vk_commit(self.internal_recursive_prover.get_dag_commit(false));
                    let input_acc_hash = poseidon2_hash_slice(&def_vk_commit).0;
                    let output_acc_hash = poseidon2_hash_slice(&[F::ZERO]).0;
                    let combined_hash =
                        poseidon2_compress_with_capacity(input_acc_hash, output_acc_hash).0;
                    Ok(DeferralProof::Absent(DeferralPvs {
                        initial_acc_hash: combined_hash,
                        final_acc_hash: combined_hash,
                        depth: F::ONE,
                    }))
                } else {
                    let mut merkle_depth = 2usize;
                    let mut layer = 0usize;

                    // Aggregate internal-for-leaf proofs down to a single proof.
                    // First pass uses DeferralAggregation (children are i4l proofs);
                    // subsequent passes use RecursiveSelf (children are ir proofs).
                    loop {
                        let is_first = layer == 0;
                        let child_merkle_depth = if proofs.len() > 1 {
                            let d = merkle_depth;
                            merkle_depth += 1;
                            Some(d)
                        } else {
                            None
                        };

                        proofs = info_span!(
                            "agg_layer",
                            group = format!("internal_recursive.{layer}"),
                            circuit = circuit_idx
                        )
                        .in_scope(|| {
                            proofs
                                .chunks(2)
                                .enumerate()
                                .map(|(idx, chunk)| {
                                    let kind = if is_first {
                                        DeferralChildVkKind::DeferralAggregation
                                    } else {
                                        DeferralChildVkKind::RecursiveSelf
                                    };
                                    info_span!("single_internal_agg", idx = idx).in_scope(|| {
                                        self.internal_recursive_prover.agg_prove::<E>(
                                            chunk,
                                            kind,
                                            child_merkle_depth,
                                        )
                                    })
                                })
                                .collect::<Result<Vec<_>>>()
                        })?;

                        layer += 1;
                        if proofs.len() == 1 {
                            break;
                        }
                    }

                    // Generate the deferral hook proof
                    Ok(DeferralProof::Present(
                        self.def_hook_prover
                            .prove::<E>(proofs.pop().unwrap(), res.leaf_io_commits)?,
                    ))
                }
            })
            .collect::<Result<Vec<_>>>()
    }

    pub fn make_extension(&self, fns: Vec<Arc<DeferralFn>>) -> DeferralExtension {
        let dag_commit = self.internal_recursive_prover.get_dag_commit(false);
        let def_vk_commits = self
            .single_circuit_provers
            .iter()
            .map(|p| {
                p.vk_commit(dag_commit)
                    .iter()
                    .flat_map(|f| f.to_unique_u32().to_le_bytes())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();
        DeferralExtension {
            fns,
            def_vk_commits,
        }
    }
}
