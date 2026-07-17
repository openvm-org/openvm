use std::sync::Arc;

use eyre::Result;
use openvm_circuit::arch::{
    instructions::exe::VmExe, Executor, MeteredExecutor, PreflightExecutor, VmBuilder,
    VmExecutionConfig,
};
use openvm_continuations::RootSC;
use openvm_stark_backend::{
    p3_field::PrimeField32, proof::Proof, prover::ProverDevice, StarkEngine, Val,
};
use openvm_verify_stark_host::VmStarkProof;

#[cfg(feature = "evm-prove")]
use crate::prover::Halo2Prover;
use crate::{
    prover::{vm::types::VmProvingKey, AggProver, InternalLayerMetadata, RootProver, StarkProver},
    DeferralInput, DeferralSetup, StdIn, SC,
};

/// EVM prover that produces a root STARK proof with Halo2 wrapping.
///
/// [`EvmProver::prove_root`] outputs the unwrapped root STARK, while
/// [`EvmProver::prove_root_from_vm_stark_proof`] outputs the unwrapped root STARK from an
/// intermediate STARK proof, for more finegrained separation of work
/// [`EvmProver::prove_evm`] produces an [`EvmProof`](crate::types::EvmProof)
/// suitable for on-chain verification.
pub struct EvmProver<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub stark_prover: StarkProver<E, VB>,
    pub root_prover: Arc<RootProver>,
    #[cfg(feature = "evm-prove")]
    pub halo2_prover: Option<Halo2Prover>,
}

impl<E, VB> EvmProver<E, VB>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E> + Clone,
    Val<SC>: PrimeField32,
{
    pub fn new(
        vm_builder: VB,
        app_vm_pk: &VmProvingKey<VB::VmConfig>,
        app_exe: Arc<VmExe<Val<SC>>>,
        agg_prover: Arc<AggProver>,
        deferral_setup: DeferralSetup,
        root_prover: Arc<RootProver>,
        #[cfg(feature = "evm-prove")] halo2_prover: Option<Halo2Prover>,
    ) -> Result<Self> {
        Ok(Self {
            stark_prover: StarkProver::new(
                vm_builder,
                app_vm_pk,
                app_exe,
                agg_prover,
                deferral_setup,
            )?,
            root_prover,
            #[cfg(feature = "evm-prove")]
            halo2_prover,
        })
    }

    pub fn prove_root_from_vm_stark_proof(
        &mut self,
        stark_proof: VmStarkProof,
        metadata: &mut InternalLayerMetadata,
    ) -> Result<Proof<RootSC>>
    where
        <VB::VmConfig as VmExecutionConfig<Val<SC>>>::Executor: Executor<Val<SC>>
            + MeteredExecutor<Val<SC>>
            + PreflightExecutor<Val<SC>, VB::RecordArena>,
    {
        #[cfg(test)]
        {
            let agg_vk = self
                .stark_prover
                .agg_prover
                .internal_recursive_prover
                .get_vk()
                .as_ref()
                .clone();
            let baseline = self.stark_prover.generate_baseline();
            crate::GenericSdk::<E, VB>::verify_proof(agg_vk, baseline, &stark_proof)?;
        }

        const MAX_ROOT_TRACEGEN_RETRIES: usize = 8;
        let agg_prover = &self.stark_prover.agg_prover;
        let root_engine = self.root_prover.create_engine();
        self.root_prover
            .prove(stark_proof, &root_engine, MAX_ROOT_TRACEGEN_RETRIES, |p| {
                agg_prover.wrap_proof(p, metadata)
            })
    }

    pub fn prove_root(
        &mut self,
        input: StdIn,
        def_inputs: &[DeferralInput],
    ) -> Result<Proof<RootSC>>
    where
        <E::PD as ProverDevice<E::PB, E::TS>>::DeviceCtx: 'static,
        <VB::VmConfig as VmExecutionConfig<Val<SC>>>::Executor: Executor<Val<SC>>
            + MeteredExecutor<Val<SC>>
            + PreflightExecutor<Val<SC>, VB::RecordArena>,
    {
        let (stark_proof, mut internal_metadata) = self.stark_prover.prove(input, def_inputs)?;
        self.prove_root_from_vm_stark_proof(stark_proof, &mut internal_metadata)
    }

    #[cfg(feature = "evm-prove")]
    pub fn prove_evm(
        &mut self,
        input: StdIn,
        def_inputs: &[DeferralInput],
    ) -> Result<crate::types::EvmProof>
    where
        <E::PD as ProverDevice<E::PB, E::TS>>::DeviceCtx: 'static,
        <VB::VmConfig as VmExecutionConfig<Val<SC>>>::Executor: Executor<Val<SC>>
            + MeteredExecutor<Val<SC>>
            + PreflightExecutor<Val<SC>, VB::RecordArena>,
    {
        let root_proof = self.prove_root(input, def_inputs)?;
        let evm_proof = self
            .halo2_prover
            .as_ref()
            .unwrap()
            .prove_for_evm(&root_proof);
        Ok(evm_proof)
    }
}
