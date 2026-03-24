use std::sync::Arc;

use eyre::Result;
use openvm_circuit::arch::{
    instructions::exe::VmExe, Executor, MeteredExecutor, PreflightExecutor, VmBuilder,
    VmExecutionConfig,
};
use openvm_continuations::RootSC;
use openvm_stark_backend::{p3_field::PrimeField32, proof::Proof, StarkEngine, Val};

#[cfg(feature = "evm-prove")]
use crate::prover::Halo2Prover;
use crate::{
    prover::{vm::types::VmProvingKey, AggProver, DeferralPathProver, RootProver, StarkProver},
    DeferralInput, StdIn, SC,
};

/// EVM prover that produces a root STARK proof with Halo2 wrapping.
///
/// [`EvmProver::prove_unwrapped`] outputs the unwrapped root STARK, while
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
        def_prover: Option<Arc<DeferralPathProver>>,
        root_prover: Arc<RootProver>,
        #[cfg(feature = "evm-prove")] halo2_prover: Option<Halo2Prover>,
    ) -> Result<Self> {
        Ok(Self {
            stark_prover: StarkProver::new(vm_builder, app_vm_pk, app_exe, agg_prover, def_prover)?,
            root_prover,
            #[cfg(feature = "evm-prove")]
            halo2_prover,
        })
    }

    pub fn prove_unwrapped(
        &mut self,
        input: StdIn<Val<SC>>,
        def_inputs: &[DeferralInput],
    ) -> Result<Proof<RootSC>>
    where
        <VB::VmConfig as VmExecutionConfig<Val<SC>>>::Executor: Executor<Val<SC>>
            + MeteredExecutor<Val<SC>>
            + PreflightExecutor<Val<SC>, VB::RecordArena>,
    {
        let (mut stark_proof, mut internal_metadata) =
            self.stark_prover.prove(input, def_inputs)?;

        let root_ctx = {
            const MAX_ROOT_TRACEGEN_RETRIES: usize = 8;
            let mut attempt = 0usize;
            loop {
                if let Some(ctx) = self.root_prover.generate_proving_ctx(stark_proof.clone()) {
                    break ctx;
                }
                if attempt >= MAX_ROOT_TRACEGEN_RETRIES {
                    return Err(eyre::eyre!(
                        "root tracegen returned None after {MAX_ROOT_TRACEGEN_RETRIES} retries"
                    ));
                }
                stark_proof = self
                    .stark_prover
                    .agg_prover
                    .wrap_proof(stark_proof, &mut internal_metadata)?;
                attempt += 1;
            }
        };

        #[cfg(test)]
        {
            for ((air_idx, air_ctx), expected_height) in root_ctx
                .per_trace
                .iter()
                .zip(self.root_prover.0.get_trace_heights().unwrap())
            {
                assert_eq!(
                    air_ctx.height(),
                    expected_height,
                    "height mismatch at {air_idx}"
                )
            }
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

        let root_proof = self.root_prover.prove_from_ctx(root_ctx)?;
        Ok(root_proof)
    }

    #[cfg(feature = "evm-prove")]
    pub fn prove_evm(
        &mut self,
        input: StdIn<Val<SC>>,
        def_inputs: &[DeferralInput],
    ) -> Result<crate::types::EvmProof>
    where
        <VB::VmConfig as VmExecutionConfig<Val<SC>>>::Executor: Executor<Val<SC>>
            + MeteredExecutor<Val<SC>>
            + PreflightExecutor<Val<SC>, VB::RecordArena>,
    {
        let root_proof = self.prove_unwrapped(input, def_inputs)?;
        let evm_proof = self
            .halo2_prover
            .as_ref()
            .unwrap()
            .prove_for_evm(&root_proof);
        Ok(evm_proof)
    }
}
