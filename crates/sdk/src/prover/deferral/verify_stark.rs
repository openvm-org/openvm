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

use crate::{
    config::{AggregationConfig, AggregationSystemParams, AggregationTreeConfig},
    keygen::{AggProvingKey, DeferralCircuitProvingKey, DeferralProvingKey},
    prover::{AggProver, DeferralPathProver, DeferralProver, SingleDefCircuitProver},
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

impl DeferralPathProver {
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
        let mut deferral_prover = DeferralProver::from_single_circuit_pks(
            Self::supported_deferral_circuit_prover_from_pk(&first_config.def_type, first_pk)?,
            deferral_pk.def_internal_recursive_pk,
            deferral_pk.def_hook_pk,
        );

        for (config, circuit_pk) in deferral_config
            .circuits
            .iter()
            .skip(1)
            .zip(deferral_circuit_pks)
        {
            deferral_prover = deferral_prover.with_single_circuit_prover(
                Self::supported_deferral_circuit_prover_from_pk(&config.def_type, circuit_pk)?,
            );
        }

        Ok(DeferralPathProver::from_pk(
            deferral_agg_pk,
            Arc::new(deferral_prover),
        ))
    }

    fn supported_deferral_circuit_prover_from_pk(
        def_type: &SupportedDeferral,
        circuit_pk: DeferralCircuitProvingKey,
    ) -> Result<SingleDefCircuitProver> {
        match def_type {
            SupportedDeferral::VerifyStark => {
                let verify_circuit_pk = circuit_pk.def_circuit_pk.as_ref().clone();
                let verify_circuit_prover =
                    <VerifyCircuitProver as DeferralCircuitProver<SC>>::from_pk(verify_circuit_pk);
                Ok(SingleDefCircuitProver::from_pks(
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

    /// Builds a [`DeferralPathProver`] backed by the verify-stark circuit, configured so an SDK
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
        let dummy_deferral_prover =
            DeferralProver::new(dummy, agg_config.clone(), hook_params.clone());

        // Construct the deferral-path AggProver, which can aggregate hook proofs from both the
        // dummy DeferralProver above and the verify-stark one below.
        let agg_prover = Arc::new(AggProver::new(
            dummy_deferral_prover.def_hook_prover.get_vk(),
            agg_config.clone(),
            AggregationTreeConfig::deferral(),
            Some(dummy_deferral_prover.def_hook_prover.get_cached_commit()),
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

        // Construct the verify-stark DeferralProver, which should have the same hook vk and cached
        // commit as the dummy one.
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
        let deferral_prover = DeferralProver::new(verify_stark_prover, agg_config, hook_params);

        assert_eq!(
            deferral_prover.def_hook_prover.get_vk().pre_hash,
            dummy_deferral_prover.def_hook_prover.get_vk().pre_hash
        );
        assert_eq!(
            deferral_prover.def_hook_prover.get_cached_commit(),
            dummy_deferral_prover.def_hook_prover.get_cached_commit()
        );

        // Return the deferral-enabled verify-stark DeferralPathProver.
        Self {
            deferral_prover: Arc::new(deferral_prover),
            agg_prover,
        }
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
