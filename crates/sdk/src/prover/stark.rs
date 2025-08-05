use std::sync::Arc;

use openvm_circuit::arch::{
    Executor, MeteredExecutor, PreflightExecutor, VirtualMachineError, VmBuilder, VmExecutionConfig,
};
use openvm_continuations::verifier::internal::types::VmStarkProof;
#[cfg(feature = "evm-prove")]
use openvm_continuations::{verifier::root::types::RootVmVerifierInput, RootSC};
use openvm_native_circuit::NativeConfig;
#[cfg(feature = "evm-prove")]
use openvm_stark_backend::proof::Proof;
use openvm_stark_sdk::engine::StarkFriEngine;

use crate::{
    config::AggregationTreeConfig,
    keygen::{AggStarkProvingKey, AppProvingKey},
    prover::{agg::AggStarkProver, app::AppProver},
    NonRootCommittedExe, StdIn, F, SC,
};

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
        app_pk: Arc<AppProvingKey<VB::VmConfig>>,
        app_committed_exe: Arc<NonRootCommittedExe>,
        agg_stark_pk: AggStarkProvingKey,
        agg_tree_config: AggregationTreeConfig,
    ) -> Result<Self, VirtualMachineError> {
        assert_eq!(
            app_pk.leaf_fri_params, agg_stark_pk.leaf_vm_pk.fri_params,
            "App VM is incompatible with Agg VM because of leaf FRI parameters"
        );
        assert_eq!(
            app_pk.app_vm_pk.vm_config.as_ref().num_public_values,
            agg_stark_pk.num_user_public_values(),
            "App VM is incompatible with Agg VM  because of the number of public values"
        );

        Ok(Self {
            app_prover: AppProver::new(
                app_vm_builder,
                app_pk.app_vm_pk.clone(),
                app_committed_exe,
            )?,
            agg_prover: AggStarkProver::new(
                native_builder,
                agg_stark_pk,
                app_pk.leaf_committed_exe.clone(),
                agg_tree_config,
            )?,
        })
    }
    pub fn set_program_name(&mut self, program_name: impl AsRef<str>) -> &mut Self {
        self.app_prover.set_program_name(program_name);
        self
    }
    #[cfg(feature = "evm-prove")]
    pub fn generate_proof_for_outer_recursion(
        &mut self,
        input: StdIn,
    ) -> Result<Proof<RootSC>, VirtualMachineError> {
        let app_proof = self.app_prover.generate_app_proof(input)?;
        self.agg_prover.generate_root_proof(app_proof)
    }
    #[cfg(feature = "evm-prove")]
    pub fn generate_root_verifier_input(
        &mut self,
        input: StdIn,
    ) -> Result<RootVmVerifierInput<SC>, VirtualMachineError> {
        let app_proof = self.app_prover.generate_app_proof(input)?;
        self.agg_prover.generate_root_verifier_input(app_proof)
    }

    pub fn generate_e2e_stark_proof(
        &mut self,
        input: StdIn,
    ) -> Result<VmStarkProof<SC>, VirtualMachineError> {
        let app_proof = self.app_prover.generate_app_proof(input)?;
        let leaf_proofs = self.agg_prover.generate_leaf_proofs(&app_proof)?;
        self.agg_prover
            .aggregate_leaf_proofs(leaf_proofs, app_proof.user_public_values.public_values)
    }
}
