use std::sync::Arc;

use eyre::Result;
use itertools::Itertools;
use openvm_continuations::{
    prover::{DeferralChildVkKind, DeferralCircuitProver},
    SC,
};
use openvm_deferral_circuit::{DeferralExtension, DeferralFn};
use openvm_recursion_circuit::utils::poseidon2_hash_slice;
use openvm_sdk_config::deferral::{DeferralCircuitConfig, DeferralConfig, SupportedDeferral};
use openvm_stark_backend::{
    keygen::types::MultiStarkProvingKey,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    proof::Proof,
    SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, DIGEST_SIZE, F,
};
use openvm_verify_stark_host::pvs::DeferralPvs;
use tracing::info_span;

use crate::{
    config::AggregationConfig,
    keygen::{AggPrefixProvingKey, DeferralCircuitProvingKey, DeferralProvingKey},
    prover::SingleDeferralCircuitProver,
    DeferralInput,
};

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

#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum DeferralProof {
    Present(Proof<SC>),
    Absent(DeferralPvs<F>),
}

/// Proves the configured set of deferral circuits through their hook proofs.
pub struct MultiDeferralCircuitProver {
    pub single_circuit_provers: Vec<SingleDeferralCircuitProver>,
    pub internal_recursive_prover: DeferralInnerProver,
    pub def_hook_prover: DeferralHookProver,
}

impl MultiDeferralCircuitProver {
    pub fn new<DP: DeferralCircuitProver<SC> + Send + Sync + 'static>(
        def_circuit_prover: DP,
        agg_config: AggregationConfig,
        hook_params: SystemParams,
    ) -> Self {
        assert_eq!(def_circuit_prover.get_def_idx(), 0);
        let single_circuit_prover = SingleDeferralCircuitProver::new(
            def_circuit_prover,
            agg_config.params.leaf,
            agg_config.params.internal.clone(),
        );
        let internal_recursive_prover = DeferralInnerProver::new::<E>(
            single_circuit_prover.internal_for_leaf_prover.get_vk(),
            agg_config.params.internal,
            true,
        );
        let internal_recursive_cached_commit = internal_recursive_prover
            .get_vk_commit(true)
            .cached_commit
            .into();
        let def_hook_prover = DeferralHookProver::new::<E>(
            internal_recursive_prover.get_vk(),
            internal_recursive_cached_commit,
            hook_params,
        );
        Self {
            single_circuit_provers: vec![single_circuit_prover],
            internal_recursive_prover,
            def_hook_prover,
        }
    }

    pub fn from_pks<DP: DeferralCircuitProver<SC> + Send + Sync + 'static>(
        def_circuit_prover: DP,
        def_prefix_pk: AggPrefixProvingKey,
        def_internal_recursive_pk: Arc<MultiStarkProvingKey<SC>>,
        def_hook_pk: Arc<MultiStarkProvingKey<SC>>,
    ) -> Self {
        assert_eq!(def_circuit_prover.get_def_idx(), 0);
        let single_circuit_prover = SingleDeferralCircuitProver::from_pks(
            def_circuit_prover,
            def_prefix_pk.leaf,
            def_prefix_pk.internal_for_leaf,
        );
        Self::from_single_circuit_prover(
            single_circuit_prover,
            def_internal_recursive_pk,
            def_hook_pk,
        )
    }

    pub fn from_single_circuit_prover(
        single_circuit_prover: SingleDeferralCircuitProver,
        def_internal_recursive_pk: Arc<MultiStarkProvingKey<SC>>,
        def_hook_pk: Arc<MultiStarkProvingKey<SC>>,
    ) -> Self {
        assert_eq!(single_circuit_prover.def_circuit_prover.get_def_idx(), 0);
        let internal_recursive_prover = DeferralInnerProver::from_pk::<E>(
            single_circuit_prover.internal_for_leaf_prover.get_vk(),
            def_internal_recursive_pk,
            true,
        );
        let internal_recursive_cached_commit = internal_recursive_prover
            .get_vk_commit(true)
            .cached_commit
            .into();
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

    pub fn with_prover<DP: DeferralCircuitProver<SC> + Send + Sync + 'static>(
        mut self,
        def_circuit_prover: DP,
    ) -> Self {
        assert_eq!(
            def_circuit_prover.get_def_idx(),
            self.single_circuit_provers.len()
        );
        let leaf_params = self.single_circuit_provers[0]
            .leaf_prover
            .get_vk()
            .inner
            .params
            .clone();
        let internal_params = self.internal_recursive_prover.get_vk().inner.params.clone();
        let single_circuit_prover =
            SingleDeferralCircuitProver::new(def_circuit_prover, leaf_params, internal_params);
        self.single_circuit_provers.push(single_circuit_prover);
        self
    }

    pub fn with_prover_from_pk<DP: DeferralCircuitProver<SC> + Send + Sync + 'static>(
        self,
        def_circuit_prover: DP,
        def_prefix_pk: AggPrefixProvingKey,
    ) -> Self {
        assert_eq!(
            def_circuit_prover.get_def_idx(),
            self.single_circuit_provers.len()
        );
        let single_circuit_prover = SingleDeferralCircuitProver::from_pks(
            def_circuit_prover,
            def_prefix_pk.leaf,
            def_prefix_pk.internal_for_leaf,
        );
        self.with_single_circuit_prover(single_circuit_prover)
    }

    pub fn with_single_circuit_prover(
        mut self,
        single_circuit_prover: SingleDeferralCircuitProver,
    ) -> Self {
        assert_eq!(
            single_circuit_prover.def_circuit_prover.get_def_idx(),
            self.single_circuit_provers.len()
        );
        self.single_circuit_provers.push(single_circuit_prover);
        self
    }

    pub fn prove(&self, inputs: &[DeferralInput]) -> Result<Vec<DeferralProof>> {
        // Generate internal-for-leaf proofs and leaf IO commits per circuit
        let per_circuit = self
            .single_circuit_provers
            .iter()
            .zip_eq(inputs)
            .map(|(prover, inputs)| prover.prove(inputs))
            .collect::<Result<Vec<_>>>()?;

        // For each circuit: do internal recursive aggregation then generate the hook proof
        let mut per_circuit = per_circuit
            .into_iter()
            .enumerate()
            .map(|(def_idx, res)| {
                let mut proofs = res.internal_for_leaf_proofs;
                if proofs.is_empty() {
                    let def_circuit_commit = self.single_circuit_provers[def_idx]
                        .circuit_commit(self.internal_recursive_prover.get_vk_commit(false));
                    Ok(DeferralProof::Absent(absent_deferral_pvs(
                        def_idx,
                        def_circuit_commit,
                    )))
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
                            circuit = def_idx
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
            .collect::<Result<Vec<_>>>()?;

        // Pad returned vector up to a power of two length with absent deferral proofs
        let target_length = per_circuit.len().next_power_of_two();
        for def_idx in per_circuit.len()..target_length {
            per_circuit.push(DeferralProof::Absent(absent_deferral_pvs(
                def_idx,
                [F::ZERO; DIGEST_SIZE],
            )));
        }

        Ok(per_circuit)
    }

    pub fn make_config(&self, supported_deferrals: Vec<SupportedDeferral>) -> DeferralConfig {
        let vk_commit = self.internal_recursive_prover.get_vk_commit(false);
        let circuits = self
            .single_circuit_provers
            .iter()
            .zip_eq(supported_deferrals)
            .map(|(p, deferral)| {
                let commit = p.circuit_commit(vk_commit).into();
                DeferralCircuitConfig {
                    def_type: deferral,
                    commit,
                }
            })
            .collect();
        DeferralConfig::new(circuits)
    }

    pub fn make_extension(&self, fns: Vec<Arc<DeferralFn>>) -> DeferralExtension {
        let vk_commit = self.internal_recursive_prover.get_vk_commit(false);
        let def_circuit_commits = self
            .single_circuit_provers
            .iter()
            .map(|p| {
                p.circuit_commit(vk_commit)
                    .iter()
                    .flat_map(|f| f.to_unique_u32().to_le_bytes())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();
        DeferralExtension {
            fns,
            def_circuit_commits,
        }
    }

    pub fn get_pk(&self) -> DeferralProvingKey {
        let circuits = self
            .single_circuit_provers
            .iter()
            .map(|single_circuit_prover| DeferralCircuitProvingKey {
                def_circuit_pk: single_circuit_prover.def_circuit_prover.get_pk(),
                agg_prefix_pk: AggPrefixProvingKey {
                    leaf: single_circuit_prover.leaf_prover.get_pk(),
                    internal_for_leaf: single_circuit_prover.internal_for_leaf_prover.get_pk(),
                },
            })
            .collect();
        DeferralProvingKey {
            circuits,
            def_internal_recursive_pk: self.internal_recursive_prover.get_pk(),
            def_hook_pk: self.def_hook_prover.get_pk(),
        }
    }
}

fn absent_deferral_pvs(def_idx: usize, def_circuit_commit: [F; DIGEST_SIZE]) -> DeferralPvs<F> {
    let input_acc_hash = poseidon2_hash_slice(&def_circuit_commit).0;
    let output_acc_hash = poseidon2_hash_slice(&[F::ZERO]).0;
    let combined_hash = poseidon2_compress_with_capacity(input_acc_hash, output_acc_hash).0;
    DeferralPvs {
        initial_acc_hash: combined_hash,
        final_acc_hash: combined_hash,
        depth: F::ONE,
        node_idx: F::from_usize(def_idx),
    }
}
