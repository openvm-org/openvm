use std::sync::Arc;

use openvm_circuit::arch::{
    instructions::exe::VmExe, Executor, MeteredExecutor, PreflightExecutor, VirtualMachineError,
    VmBuilder, VmExecutionConfig,
};
use openvm_continuations::verifier::internal::types::VmStarkProof;
#[cfg(feature = "evm-prove")]
use openvm_continuations::{verifier::root::types::RootVmVerifierInput, RootSC};
use openvm_native_circuit::NativeConfig;
#[cfg(feature = "evm-prove")]
use openvm_stark_backend::proof::Proof;
use openvm_stark_sdk::engine::StarkFriEngine;

use crate::{
    commit::AppExecutionCommit,
    config::AggregationTreeConfig,
    keygen::{AggProvingKey, AppProvingKey},
    prover::{agg::AggStarkProver, app::AppProver},
    StdIn, F, SC,
};

/// This prover contains an [`app_prover`](StarkProver::app_prover) internally.
pub struct StarkProver<E, VB, NativeBuilder>
where
    E: StarkFriEngine<SC = SC>,
    VB: VmBuilder<E>,
    NativeBuilder: VmBuilder<E, VmConfig = NativeConfig>,
{
    pub app_prover: AppProver<E, VB>,
    pub agg_prover: AggStarkProver<E, NativeBuilder>,
}
impl<E, VB, NativeBuilder> StarkProver<E, VB, NativeBuilder>
where
    E: StarkFriEngine<SC = SC>,
    VB: VmBuilder<E>,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor:
        Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, <VB as VmBuilder<E>>::RecordArena>,
    NativeBuilder: VmBuilder<E, VmConfig = NativeConfig> + Clone,
    <NativeConfig as VmExecutionConfig<F>>::Executor:
        PreflightExecutor<F, <NativeBuilder as VmBuilder<E>>::RecordArena>,
{
    pub fn new(
        app_vm_builder: VB,
        native_builder: NativeBuilder,
        app_pk: &AppProvingKey<VB::VmConfig>,
        app_exe: Arc<VmExe<F>>,
        agg_pk: &AggProvingKey,
        agg_tree_config: AggregationTreeConfig,
    ) -> Result<Self, VirtualMachineError> {
        assert_eq!(
            app_pk.leaf_fri_params, agg_pk.leaf_vm_pk.fri_params,
            "App VM is incompatible with Agg VM because of leaf FRI parameters"
        );
        assert_eq!(
            app_pk.app_vm_pk.vm_config.as_ref().num_public_values,
            agg_pk.num_user_public_values(),
            "App VM is incompatible with Agg VM  because of the number of public values"
        );

        Ok(Self {
            app_prover: AppProver::new(
                app_vm_builder,
                &app_pk.app_vm_pk,
                app_exe,
                app_pk.leaf_committed_exe.get_program_commit(),
            )?,
            agg_prover: AggStarkProver::new(
                native_builder,
                agg_pk,
                app_pk.leaf_committed_exe.exe.clone(),
                agg_tree_config,
            )?,
        })
    }

    pub fn from_parts(
        app_prover: AppProver<E, VB>,
        agg_prover: AggStarkProver<E, NativeBuilder>,
    ) -> Result<Self, VirtualMachineError> {
        Ok(Self {
            app_prover,
            agg_prover,
        })
    }

    pub fn with_program_name(mut self, program_name: impl AsRef<str>) -> Self {
        self.set_program_name(program_name);
        self
    }
    pub fn set_program_name(&mut self, program_name: impl AsRef<str>) -> &mut Self {
        self.app_prover.set_program_name(program_name);
        self
    }

    pub fn app_commit(&self) -> AppExecutionCommit {
        self.app_prover.app_commit()
    }

    pub fn prove(&mut self, input: StdIn) -> Result<VmStarkProof<SC>, VirtualMachineError> {
        let app_proof = self.app_prover.prove(input)?;
        let leaf_proofs = self.agg_prover.generate_leaf_proofs(&app_proof)?;
        self.agg_prover
            .aggregate_leaf_proofs(leaf_proofs, app_proof.user_public_values.public_values)
    }

    #[cfg(feature = "evm-prove")]
    pub fn generate_proof_for_outer_recursion(
        &mut self,
        input: StdIn,
    ) -> Result<Proof<RootSC>, VirtualMachineError> {
        let app_proof = self.app_prover.prove(input)?;
        self.agg_prover.generate_root_proof(app_proof)
    }
    #[cfg(feature = "evm-prove")]
    pub fn generate_root_verifier_input(
        &mut self,
        input: StdIn,
    ) -> Result<RootVmVerifierInput<SC>, VirtualMachineError> {
        let app_proof = self.app_prover.prove(input)?;
        self.agg_prover.generate_root_verifier_input(app_proof)
    }
}
