use std::sync::Arc;

use eyre::{eyre, Result};
use openvm_circuit::system::memory::dimensions::MemoryDimensions;
use openvm_continuations::{
    circuit::deferral::dummy::dummy_deferral_circuit_vk,
    prover::{DeferralCircuitProver, DeferralCircuitProverKey},
    SC,
};
use openvm_sdk_config::deferral::{DeferralConfig, SupportedDeferral};
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, proof::Proof, SystemParams};
use openvm_stark_sdk::config::baby_bear_poseidon2::Digest;

use crate::{
    config::{AggregationConfig, AggregationSystemParams, AggregationTreeConfig},
    keygen::{AggPrefixProvingKey, AggProvingKey, DeferralCircuitProvingKey, DeferralProvingKey},
    prover::{AggProver, MultiDeferralCircuitProver, SingleDeferralCircuitProver},
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_verify_stark_circuit::prover::DeferredVerifyGpuProver as VerifyProver;
        use openvm_verify_stark_circuit::prover::DeferredVerifyGpuCircuitProver as VerifyCircuitProver;
        type E = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
    } else {
        use openvm_verify_stark_circuit::prover::DeferredVerifyCpuProver as VerifyProver;
        use openvm_verify_stark_circuit::prover::DeferredVerifyCpuCircuitProver as VerifyCircuitProver;
        type E = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
    }
}

pub struct DeferralAggProver {
    pub multi_deferral_circuit_prover: Arc<MultiDeferralCircuitProver>,
    pub agg_prover: Arc<AggProver>,
}

impl DeferralAggProver {
    pub fn get_pk(&self) -> AggProvingKey {
        AggProvingKey {
            prefix: AggPrefixProvingKey {
                leaf: self.agg_prover.leaf_prover.get_pk(),
                internal_for_leaf: self.agg_prover.internal_for_leaf_prover.get_pk(),
            },
            internal_recursive: self.agg_prover.internal_recursive_prover.get_pk(),
        }
    }

    pub fn new(
        agg_config: AggregationConfig,
        multi_deferral_circuit_prover: Arc<MultiDeferralCircuitProver>,
    ) -> Self {
        let agg_prover = AggProver::new(
            multi_deferral_circuit_prover.def_hook_prover.get_vk(),
            agg_config,
            AggregationTreeConfig::deferral(),
            Some(
                multi_deferral_circuit_prover
                    .def_hook_prover
                    .get_cached_commit(),
            ),
        );
        DeferralAggProver {
            multi_deferral_circuit_prover,
            agg_prover: Arc::new(agg_prover),
        }
    }

    pub fn from_pk(
        pk: AggProvingKey,
        multi_deferral_circuit_prover: Arc<MultiDeferralCircuitProver>,
    ) -> DeferralAggProver {
        let agg_prover = AggProver::from_pk(
            multi_deferral_circuit_prover.def_hook_prover.get_vk(),
            pk,
            AggregationTreeConfig::deferral(),
            Some(
                multi_deferral_circuit_prover
                    .def_hook_prover
                    .get_cached_commit(),
            ),
        );
        DeferralAggProver {
            multi_deferral_circuit_prover,
            agg_prover: Arc::new(agg_prover),
        }
    }

    pub fn def_hook_cached_commit(&self) -> Digest {
        self.multi_deferral_circuit_prover
            .def_hook_prover
            .get_cached_commit()
    }

    pub fn def_hook_commit(&self) -> Digest {
        self.agg_prover.vm_or_hook_commit()
    }

    /// Reconstructs a [`DeferralAggProver`] from cached SDK proving keys for deferral circuits
    /// whose prover type is known to the SDK.
    ///
    /// The cached app VM config supplies the ordered [`DeferralConfig`], including each supported
    /// deferral kind and the commit that the guest VM expects. The cached deferral proving key
    /// supplies the matching circuit proving material, plus the shared internal-recursive and hook
    /// proving keys. This constructor stitches those pieces back into a
    /// [`MultiDeferralCircuitProver`] and then restores the deferral aggregation prover from
    /// `deferral_agg_pk`.
    ///
    /// Custom deferral circuits cannot be reconstructed here because the SDK does not know their
    /// concrete prover types; callers should manually build a [`MultiDeferralCircuitProver`] or
    /// [`DeferralAggProver`] for those cases.
    pub(crate) fn from_supported_deferral_pks(
        deferral_config: &DeferralConfig,
        deferral_pk: DeferralProvingKey,
        deferral_agg_pk: AggProvingKey,
    ) -> Result<Self> {
        if deferral_config.circuits.len() != deferral_pk.circuits.len() {
            return Err(eyre!(
                "cached deferral proving key circuit count does not match app VM deferral config"
            ));
        }

        let mut deferral_circuit_pks = deferral_pk.circuits.into_iter();
        let first_config = deferral_config
            .circuits
            .first()
            .ok_or_else(|| eyre!("app VM deferral config has no circuits"))?;
        let first_pk = deferral_circuit_pks
            .next()
            .ok_or_else(|| eyre!("cached deferral proving key has no circuits"))?;
        let mut multi_deferral_circuit_prover = MultiDeferralCircuitProver::from_single_circuit_pks(
            supported_deferral_circuit_prover_from_pk(&first_config.def_type, first_pk)?,
            deferral_pk.def_internal_recursive_pk,
            deferral_pk.def_hook_pk,
        );

        for (config, circuit_pk) in deferral_config
            .circuits
            .iter()
            .skip(1)
            .zip(deferral_circuit_pks)
        {
            multi_deferral_circuit_prover =
                multi_deferral_circuit_prover.with_single_circuit_prover(
                    supported_deferral_circuit_prover_from_pk(&config.def_type, circuit_pk)?,
                );
        }

        #[cfg(debug_assertions)]
        {
            let reconstructed_config = multi_deferral_circuit_prover.make_config(
                deferral_config
                    .circuits
                    .iter()
                    .map(|circuit| circuit.def_type.clone())
                    .collect(),
            );
            for (expected, reconstructed) in deferral_config
                .circuits
                .iter()
                .zip(&reconstructed_config.circuits)
            {
                assert_eq!(
                    reconstructed, expected,
                    "cached deferral proving key circuit config does not match app VM deferral config"
                );
            }
        }

        Ok(DeferralAggProver::from_pk(
            deferral_agg_pk,
            Arc::new(multi_deferral_circuit_prover),
        ))
    }

    /// Builds a [`DeferralAggProver`] backed by the verify-stark circuit, configured so an SDK
    /// with the given params can recursively verify the VM STARK proofs it produces, including its
    /// own deferral-carrying proofs.
    ///
    /// The deferral-enabled internal-recursive vk and the self-referential `def_hook_commit` are
    /// derived internally from a dummy deferral circuit.
    pub fn verify_stark(
        agg_params: &AggregationSystemParams,
        hook_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
    ) -> Self {
        // Derive the deferral path's fixed-point artifacts with a cheap dummy deferral circuit.
        let dummy = DummyDefCircuitProver {
            vk: dummy_deferral_circuit_vk::<E>(agg_params.internal.clone()),
        };
        let agg_config = AggregationConfig {
            params: agg_params.clone(),
        };
        let dummy_multi_deferral_circuit_prover =
            MultiDeferralCircuitProver::new(dummy, agg_config.clone(), hook_params.clone());

        // Construct the deferral-path AggProver, which can aggregate hook proofs from both the
        // dummy MultiDeferralCircuitProver above and the verify-stark one below.
        let agg_prover = Arc::new(AggProver::new(
            dummy_multi_deferral_circuit_prover.def_hook_prover.get_vk(),
            agg_config.clone(),
            AggregationTreeConfig::deferral(),
            Some(
                dummy_multi_deferral_circuit_prover
                    .def_hook_prover
                    .get_cached_commit(),
            ),
        ));

        // The deferral-path aggregation tree's internal-recursive vk is a universal copy of the VM
        // internal-recursive vk that a verify-stark circuit verifies.
        let ir_vk = agg_prover.internal_recursive_prover.get_vk();
        let ir_cached_commit = agg_prover
            .internal_recursive_prover
            .get_self_vk_pcs_data()
            .expect("internal-recursive prover must expose its self vk pcs data")
            .commitment
            .into();
        let def_hook_commit = agg_prover.vm_or_hook_commit();

        // Construct the verify-stark MultiDeferralCircuitProver, which should have the same hook vk
        // and cached commit as the dummy one.
        let deferred_verify_prover = VerifyProver::new::<E>(
            ir_vk,
            ir_cached_commit,
            agg_params.internal.clone(),
            memory_dimensions,
            num_user_pvs,
            Some(def_hook_commit.into()),
            0,
        );
        let verify_stark_prover = VerifyCircuitProver::new(deferred_verify_prover);
        let multi_deferral_circuit_prover =
            MultiDeferralCircuitProver::new(verify_stark_prover, agg_config, hook_params);

        assert_eq!(
            multi_deferral_circuit_prover
                .def_hook_prover
                .get_vk()
                .pre_hash,
            dummy_multi_deferral_circuit_prover
                .def_hook_prover
                .get_vk()
                .pre_hash
        );
        assert_eq!(
            multi_deferral_circuit_prover
                .def_hook_prover
                .get_cached_commit(),
            dummy_multi_deferral_circuit_prover
                .def_hook_prover
                .get_cached_commit()
        );

        // Return the deferral-enabled verify-stark DeferralAggProver.
        Self {
            multi_deferral_circuit_prover: Arc::new(multi_deferral_circuit_prover),
            agg_prover,
        }
    }
}

fn supported_deferral_circuit_prover_from_pk(
    def_type: &SupportedDeferral,
    circuit_pk: DeferralCircuitProvingKey,
) -> Result<SingleDeferralCircuitProver> {
    match def_type {
        SupportedDeferral::VerifyStark => {
            let verify_circuit_pk = circuit_pk.def_circuit_pk.as_ref().clone();
            let verify_circuit_prover =
                <VerifyCircuitProver as DeferralCircuitProver<SC>>::from_pk(verify_circuit_pk);
            Ok(SingleDeferralCircuitProver::from_pks(
                verify_circuit_prover,
                circuit_pk.agg_prefix_pk.leaf,
                circuit_pk.agg_prefix_pk.internal_for_leaf,
            ))
        }
        SupportedDeferral::Other(name) => Err(eyre!(
            "custom deferral circuit provers need to be manually created for deferral `{name}`"
        )),
    }
}

/// A dummy [`DeferralCircuitProver`] that only exposes a trivial verifying key. It exists solely to
/// seed the deferral aggregation chain when deriving the deferral path fixed point; its `prove`
/// method is never called.
struct DummyDefCircuitProver {
    vk: Arc<MultiStarkVerifyingKey<SC>>,
}

impl DeferralCircuitProver<SC> for DummyDefCircuitProver {
    fn get_pk(&self) -> Arc<DeferralCircuitProverKey<SC>> {
        unreachable!("DummyDefCircuitProver does not have proving material")
    }

    fn from_pk(_encoded_pk: DeferralCircuitProverKey<SC>) -> Self
    where
        Self: Sized,
    {
        unreachable!("DummyDefCircuitProver does not have proving material")
    }

    fn get_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        self.vk.clone()
    }

    fn prove(&self, _input_bytes: &[u8]) -> Proof<SC> {
        unreachable!("DummyDefCircuitProver is only used to derive deferral path artifacts")
    }

    fn get_def_idx(&self) -> usize {
        0
    }
}
