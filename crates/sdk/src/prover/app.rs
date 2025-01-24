use std::sync::Arc;

use getset::Getters;
use openvm_circuit::arch::VmConfig;
use openvm_stark_backend::{proof::Proof, Chip};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Engine;
use tracing::info_span;

use super::vm::SingleSegmentVmProver;
use crate::{
    prover::vm::{
        local::VmLocalProver, types::VmProvingKey, ContinuationVmProof, ContinuationVmProver,
    },
    NonRootCommittedExe, StdIn, F, SC,
};

#[derive(Getters)]
pub struct AppProver<VC> {
    pub program_name: Option<String>,
    #[getset(get = "pub")]
    app_prover: VmLocalProver<SC, VC, BabyBearPoseidon2Engine>,
}

impl<VC> AppProver<VC> {
    pub fn new(
        app_vm_pk: Arc<VmProvingKey<SC, VC>>,
        app_committed_exe: Arc<NonRootCommittedExe>,
    ) -> Self
    where
        VC: VmConfig<F>,
    {
        Self {
            program_name: None,
            app_prover: VmLocalProver::<SC, VC, BabyBearPoseidon2Engine>::new(
                app_vm_pk,
                app_committed_exe,
            ),
        }
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
    pub fn generate_app_proof(&self, input: StdIn) -> ContinuationVmProof<SC>
    where
        VC: VmConfig<F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        assert!(
            self.vm_config().system().continuation_enabled,
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
            #[cfg(feature = "bench-metrics")]
            metrics::counter!("fri.log_blowup")
                .absolute(self.app_prover.pk.fri_params.log_blowup as u64);
            ContinuationVmProver::prove(&self.app_prover, input)
        })
    }

    pub fn generate_app_proof_without_continuations(&self, input: StdIn) -> Proof<SC>
    where
        VC: VmConfig<F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        assert!(
            !self.vm_config().system().continuation_enabled,
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
            #[cfg(feature = "bench-metrics")]
            metrics::counter!("fri.log_blowup")
                .absolute(self.app_prover.pk.fri_params.log_blowup as u64);
            SingleSegmentVmProver::prove(&self.app_prover, input)
        })
    }

    /// App VM config
    pub fn vm_config(&self) -> &VC {
        self.app_prover.vm_config()
    }
}
