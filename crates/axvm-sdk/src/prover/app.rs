use std::sync::Arc;

use ax_stark_backend::Chip;
use ax_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Engine;
#[cfg(feature = "bench-metrics")]
use axvm_circuit::arch::{instructions::exe::AxVmExe, VmExecutor};
use axvm_circuit::{
    arch::VmConfig,
    prover::{
        local::VmLocalProver, types::VmProvingKey, ContinuationVmProof, ContinuationVmProver,
    },
};

use crate::{NonRootCommittedExe, StdIn, F, SC};

pub struct AppProver<VC> {
    app_prover: VmLocalProver<SC, VC, BabyBearPoseidon2Engine>,
}

impl<VC> AppProver<VC> {
    pub fn new(app_vm_pk: VmProvingKey<SC, VC>, app_committed_exe: Arc<NonRootCommittedExe>) -> Self
    where
        VC: VmConfig<F>,
    {
        Self {
            app_prover: VmLocalProver::<SC, VC, BabyBearPoseidon2Engine>::new(
                app_vm_pk,
                app_committed_exe,
            ),
        }
    }
    pub fn generate_app_proof(&self, input: StdIn) -> ContinuationVmProof<SC>
    where
        VC: VmConfig<F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        #[cfg(feature = "bench-metrics")]
        emit_app_execution_metrics(
            self.app_prover.pk.vm_config.clone(),
            self.app_prover.committed_exe.exe.clone(),
            input.clone(),
        );
        ContinuationVmProver::prove(&self.app_prover, input)
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
