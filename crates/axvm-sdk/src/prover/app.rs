use std::sync::Arc;

use ax_stark_backend::Chip;
use ax_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Engine;
use axvm_circuit::arch::VmConfig;
#[cfg(feature = "bench-metrics")]
use axvm_circuit::arch::{instructions::exe::AxVmExe, VmExecutor};
use tracing::info_span;

use crate::{
    prover::vm::{
        local::VmLocalProver, types::VmProvingKey, ContinuationVmProof, ContinuationVmProver,
    },
    NonRootCommittedExe, StdIn, F, SC,
};

pub struct AppProver<VC> {
    /// If true, will run execution once with full metric collection for
    /// flamegraphs (WARNING: this degrades performance).
    pub profile: bool,
    pub program_name: Option<String>,
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
            profile: false,
            program_name: None,
            app_prover: VmLocalProver::<SC, VC, BabyBearPoseidon2Engine>::new(
                app_vm_pk,
                app_committed_exe,
            ),
        }
    }

    pub fn with_profile(mut self) -> Self {
        self.profile = true;
        self
    }

    pub fn set_program_name(&mut self, program_name: impl AsRef<str>) -> &mut Self {
        self.program_name = Some(program_name.as_ref().to_string());
        self
    }

    pub fn with_program_name(mut self, program_name: impl AsRef<str>) -> Self {
        self.set_program_name(program_name);
        self
    }

    pub fn generate_app_proof(&self, input: StdIn) -> ContinuationVmProof<SC>
    where
        VC: VmConfig<F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        info_span!(
            "app proof",
            group = self
                .program_name
                .as_ref()
                .unwrap_or(&"app_proof".to_string())
        )
        .in_scope(|| {
            #[cfg(feature = "bench-metrics")]
            if self.profile {
                emit_app_execution_metrics(
                    self.app_prover.pk.vm_config.clone(),
                    self.app_prover.committed_exe.exe.clone(),
                    input.clone(),
                );
            }
            #[cfg(feature = "bench-metrics")]
            metrics::counter!("fri.log_blowup")
                .absolute(self.app_prover.pk.fri_params.log_blowup as u64);
            ContinuationVmProver::prove(&self.app_prover, input)
        })
    }
}

#[cfg(feature = "bench-metrics")]
fn emit_app_execution_metrics<VC: VmConfig<F>>(mut vm_config: VC, exe: AxVmExe<F>, input: StdIn)
where
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    vm_config.system_mut().collect_metrics = true;
    let vm = VmExecutor::new(vm_config);
    vm.execute_segments(exe, input).unwrap();
}
