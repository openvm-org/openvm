use std::sync::Arc;

use getset::Getters;
use openvm_circuit::{
    arch::{
        ContinuationVmProof, ContinuationVmProver, InsExecutorE1, InsExecutorE2,
        InstructionExecutor, SingleSegmentVmProver, VirtualMachineError, VmBuilder,
        VmExecutionConfig, VmLocalProver,
    },
    system::program::trace::VmCommittedExe,
};
use openvm_stark_backend::{config::Val, p3_field::PrimeField32, proof::Proof};
use openvm_stark_sdk::engine::{StarkEngine, StarkFriEngine};
use tracing::info_span;

use crate::{
    prover::vm::{new_local_prover, types::VmProvingKey},
    StdIn,
};

#[derive(Getters)]
pub struct AppProver<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub program_name: Option<String>,
    #[getset(get = "pub")]
    app_prover: VmLocalProver<E, VB>,
}

impl<E, VB> AppProver<E, VB>
where
    E: StarkFriEngine,
    Val<E::SC>: PrimeField32,
    VB: VmBuilder<E>,
{
    pub fn new(
        vm_builder: VB,
        app_vm_pk: Arc<VmProvingKey<E::SC, VB::VmConfig>>,
        app_committed_exe: Arc<VmCommittedExe<E::SC>>,
    ) -> Result<Self, VirtualMachineError> {
        let app_prover = new_local_prover(vm_builder, &app_vm_pk, &app_committed_exe)?;
        Ok(Self {
            program_name: None,
            app_prover,
        })
    }
    pub fn set_program_name(&mut self, program_name: impl AsRef<str>) -> &mut Self {
        self.program_name = Some(program_name.as_ref().to_string());
        self
    }
    pub fn with_program_name(mut self, program_name: impl AsRef<str>) -> Self {
        self.set_program_name(program_name);
        self
    }

    /// Generates proof for every continuation segment
    pub fn generate_app_proof(
        &mut self,
        input: StdIn<Val<E::SC>>,
    ) -> Result<ContinuationVmProof<E::SC>, VirtualMachineError>
    where
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: InsExecutorE1<Val<E::SC>>
            + InsExecutorE2<Val<E::SC>>
            + InstructionExecutor<Val<E::SC>, VB::RecordArena>,
    {
        assert!(
            self.vm_config().as_ref().continuation_enabled,
            "Use generate_app_proof_without_continuations instead."
        );
        info_span!(
            "app proof",
            group = self
                .program_name
                .as_ref()
                .unwrap_or(&"app_proof".to_string())
        )
        .in_scope(|| {
            #[cfg(feature = "metrics")]
            metrics::counter!("fri.log_blowup")
                .absolute(self.app_prover.vm.engine.fri_params().log_blowup as u64);
            ContinuationVmProver::prove(&mut self.app_prover, input)
        })
    }

    pub fn generate_app_proof_without_continuations(
        &mut self,
        input: StdIn<Val<E::SC>>,
        trace_heights: &[u32],
    ) -> Result<Proof<E::SC>, VirtualMachineError>
    where
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
            InstructionExecutor<Val<E::SC>, VB::RecordArena>,
    {
        assert!(
            !self.vm_config().as_ref().continuation_enabled,
            "Use generate_app_proof instead."
        );
        info_span!(
            "app proof",
            group = self
                .program_name
                .as_ref()
                .unwrap_or(&"app_proof".to_string())
        )
        .in_scope(|| {
            #[cfg(feature = "metrics")]
            metrics::counter!("fri.log_blowup")
                .absolute(self.app_prover.vm.engine.fri_params().log_blowup as u64);
            SingleSegmentVmProver::prove(&mut self.app_prover, input, trace_heights)
        })
    }

    /// App VM config
    pub fn vm_config(&self) -> &VB::VmConfig {
        self.app_prover.vm.config()
    }
}
